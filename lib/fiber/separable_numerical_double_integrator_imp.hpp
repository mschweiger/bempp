#include "separable_numerical_double_integrator.hpp" // To keep IDEs happy

#include "array_2d.hpp"
#include "array_3d.hpp"
#include "array_4d.hpp"

#include "basis.hpp"
#include "basis_data.hpp"
#include "expression.hpp"
#include "geometrical_data.hpp"
#include "kernel.hpp"
#include "opencl_options.hpp"
#include "opencl_framework.hpp"
#include "types.hpp"
#include "CL/separable_numerical_double_integrator.cl.str"

#include <cassert>
#include <memory>

namespace Fiber
{

template <typename ValueType, typename GeometryFactory>
SeparableNumericalDoubleIntegrator<ValueType, GeometryFactory>::
SeparableNumericalDoubleIntegrator(
        const arma::Mat<ValueType>& localTestQuadPoints,
        const arma::Mat<ValueType>& localTrialQuadPoints,
        const std::vector<ValueType> testQuadWeights,
        const std::vector<ValueType> trialQuadWeights,
        const GeometryFactory& geometryFactory,
        const arma::Mat<ValueType>& vertices,
        const arma::Mat<int>& elementCornerIndices,
        const arma::Mat<char>& auxElementData,
        const Expression<ValueType>& testExpression,
        const Kernel<ValueType>& kernel,
        const Expression<ValueType>& trialExpression,
	const OpenClFramework<ValueType,int> *openClFramework,
        const OpenClOptions& openClOptions) :
    m_localTestQuadPoints(localTestQuadPoints),
    m_localTrialQuadPoints(localTrialQuadPoints),
    m_testQuadWeights(testQuadWeights),
    m_trialQuadWeights(trialQuadWeights),
    m_geometryFactory(geometryFactory),
    m_vertices(vertices),
    m_elementCornerIndices(elementCornerIndices),
    m_auxElementData(auxElementData),
    m_testExpression(testExpression),
    m_kernel(kernel),
    m_trialExpression(trialExpression),
    m_openClFramework(openClFramework),
    m_openClOptions(openClOptions)
{
    if (localTestQuadPoints.n_cols != testQuadWeights.size())
        throw std::invalid_argument("SeparableNumericalDoubleIntegrator::"
                                    "SeparableNumericalDoubleIntegrator(): "
                                    "numbers of test points and weight do not match");
    if (localTrialQuadPoints.n_cols != trialQuadWeights.size())
        throw std::invalid_argument("SeparableNumericalDoubleIntegrator::"
                                    "SeparableNumericalDoubleIntegrator(): "
                                    "numbers of trial points and weight do not match");
    // push geometry to CL device
    if (openClFramework) {
        openClFramework->pushGeometry (vertices, elementCornerIndices);  // move this to integration manager
	clTestQuadPoints = openClFramework->pushValueMatrix (localTestQuadPoints);
	clTrialQuadPoints = openClFramework->pushValueMatrix (localTrialQuadPoints);
    }
}

template <typename ValueType, typename GeometryFactory>
SeparableNumericalDoubleIntegrator<ValueType, GeometryFactory>::
~SeparableNumericalDoubleIntegrator()
{
    if (m_openClFramework)
    {
        delete clTestQuadPoints;
	delete clTrialQuadPoints;
    }
}

template <typename ValueType, typename GeometryFactory>
inline void SeparableNumericalDoubleIntegrator<ValueType, GeometryFactory>::
setupGeometryConveniently(
        int elementIndex, typename GeometryFactory::Geometry& geometry) const
{
    setupGeometry(elementIndex,
                  m_vertices, m_elementCornerIndices, m_auxElementData,
                  geometry);
}

template <typename ValueType, typename GeometryFactory>
void SeparableNumericalDoubleIntegrator<ValueType, GeometryFactory>::integrate(
        CallVariant callVariant,
        const std::vector<int>& elementIndicesA,
        int elementIndexB,
        const Basis<ValueType>& basisA,
        const Basis<ValueType>& basisB,
        LocalDofIndex localDofIndexB,
        arma::Cube<ValueType>& result) const
{
    if (m_openClFramework)
        integrateCl (m_openClFramework, callVariant, elementIndicesA, elementIndexB, basisA, basisB,
		     localDofIndexB, result);
    else
        integrateCpu (callVariant, elementIndicesA, elementIndexB, basisA, basisB, localDofIndexB, result);
}

template <typename ValueType, typename GeometryFactory>
void SeparableNumericalDoubleIntegrator<ValueType, GeometryFactory>::integrateCpu(
        CallVariant callVariant,
        const std::vector<int>& elementIndicesA,
        int elementIndexB,
        const Basis<ValueType>& basisA,
        const Basis<ValueType>& basisB,
        LocalDofIndex localDofIndexB,
        arma::Cube<ValueType>& result) const
{
    const int testPointCount = m_localTestQuadPoints.n_cols;
    const int trialPointCount = m_localTrialQuadPoints.n_cols;
    const int elementACount = elementIndicesA.size();

    if (testPointCount == 0 || trialPointCount == 0 || elementACount == 0)
        return;
    // TODO: in the (pathological) case that pointCount == 0 but
    // geometryCount != 0, set elements of result to 0.

    // Evaluate constants
    const int testComponentCount = m_testExpression.codomainDimension();
    const int trialComponentCount = m_trialExpression.codomainDimension();
    const int dofCountA = basisA.size();
    const int dofCountB = localDofIndexB == ALL_DOFS ? basisB.size() : 1;
    const int testDofCount = callVariant == TEST_TRIAL ? dofCountA : dofCountB;
    const int trialDofCount = callVariant == TEST_TRIAL ? dofCountB : dofCountA;

    const int kernelRowCount = m_kernel.codomainDimension();
    const int kernelColCount = m_kernel.domainDimension();

    // Assert that the kernel tensor dimensions are compatible
    // with the number of components of the functions

    // TODO: This will need to be modified once we allow scalar-valued kernels
    // (treated as if they were multiplied by the unit tensor) with
    // vector-valued functions
    assert(testComponentCount == kernelRowCount);
    assert(kernelColCount == trialComponentCount);

    BasisData<ValueType> testBasisData, trialBasisData;
    GeometricalData<ValueType> testGeomData, trialGeomData;

    int testBasisDeps = 0, trialBasisDeps = 0;
    int testGeomDeps = INTEGRATION_ELEMENTS;
    int trialGeomDeps = INTEGRATION_ELEMENTS;

    m_testExpression.addDependencies(testBasisDeps, testGeomDeps);
    m_trialExpression.addDependencies(trialBasisDeps, trialGeomDeps);
    m_kernel.addGeometricalDependencies(testGeomDeps, trialGeomDeps);

    typedef typename GeometryFactory::Geometry Geometry;
    std::auto_ptr<Geometry> geometryA(m_geometryFactory.make());
    std::auto_ptr<Geometry> geometryB(m_geometryFactory.make());

    arma::Cube<ValueType> testValues, trialValues;
    Array4D<ValueType> kernelValues(kernelRowCount, kernelColCount,
                                    testPointCount, trialPointCount);

    result.set_size(testDofCount, trialDofCount, elementACount);

    setupGeometryConveniently(elementIndexB, *geometryB);
    if (callVariant == TEST_TRIAL)
    {
        basisA.evaluate(testBasisDeps, m_localTestQuadPoints, ALL_DOFS, testBasisData);
        basisB.evaluate(trialBasisDeps, m_localTrialQuadPoints, localDofIndexB, trialBasisData);
        geometryB->getData(trialGeomDeps, m_localTrialQuadPoints, trialGeomData);
        m_trialExpression.evaluate(trialBasisData, trialGeomData, trialValues);
    }
    else
    {
        basisA.evaluate(trialBasisDeps, m_localTrialQuadPoints, ALL_DOFS, trialBasisData);
        basisB.evaluate(testBasisDeps, m_localTestQuadPoints, localDofIndexB, testBasisData);
        geometryB->getData(testGeomDeps, m_localTestQuadPoints, testGeomData);
        m_testExpression.evaluate(testBasisData, testGeomData, testValues);
    }

    // Iterate over the elements
    for (int indexA = 0; indexA < elementACount; ++indexA)
    {
        setupGeometryConveniently(elementIndicesA[indexA], *geometryA);
        if (callVariant == TEST_TRIAL)
        {
            geometryA->getData(testGeomDeps, m_localTestQuadPoints, testGeomData);
            m_testExpression.evaluate(testBasisData, testGeomData, testValues);
        }
        else
        {
            geometryA->getData(trialGeomDeps, m_localTrialQuadPoints, trialGeomData);
            m_trialExpression.evaluate(trialBasisData, trialGeomData, trialValues);
        }

        m_kernel.evaluateOnGrid(testGeomData, trialGeomData, kernelValues);

        // For now, we assume that the kernel is (general) tensorial,
        // later we might handle specially the case of it being a scalar
        // times the identity tensor.
        for (int trialDof = 0; trialDof < trialDofCount; ++trialDof)
            for (int testDof = 0; testDof < testDofCount; ++testDof)
            {
                ValueType sum = 0.;
                for (int trialPoint = 0; trialPoint < trialPointCount; ++trialPoint)
                    for (int testPoint = 0; testPoint < testPointCount; ++testPoint)
                        for (int trialDim = 0; trialDim < trialComponentCount; ++trialDim)
                            for (int testDim = 0; testDim < testComponentCount; ++testDim)
                                sum +=  m_testQuadWeights[testPoint] *
                                        testGeomData.integrationElements(testPoint) *
                                        testValues(testDim, testDof, testPoint) *
                                        kernelValues(testDim, trialDim, testPoint, trialPoint) *
                                        trialValues(trialDim, trialDof, trialPoint) *
                                        trialGeomData.integrationElements(trialPoint) *
                                        m_trialQuadWeights[trialPoint];
                result(testDof, trialDof, indexA) = sum;
            }
    }
}

template <typename ValueType, typename GeometryFactory>
void SeparableNumericalDoubleIntegrator<ValueType, GeometryFactory>::integrate(
            const std::vector<ElementIndexPair>& elementIndexPairs,
            const Basis<ValueType>& testBasis,
            const Basis<ValueType>& trialBasis,
            arma::Cube<ValueType>& result) const
{
    const int testPointCount = m_localTestQuadPoints.n_cols;
    const int trialPointCount = m_localTrialQuadPoints.n_cols;
    const int geometryPairCount = elementIndexPairs.size();

    if (testPointCount == 0 || trialPointCount == 0 || geometryPairCount == 0)
        return;
    // TODO: in the (pathological) case that pointCount == 0 but
    // geometryPairCount != 0, set elements of result to 0.

    // Evaluate constants
    const int testComponentCount = m_testExpression.codomainDimension();
    const int trialComponentCount = m_trialExpression.codomainDimension();
    const int testDofCount = testBasis.size();
    const int trialDofCount = trialBasis.size();

    const int kernelRowCount = m_kernel.codomainDimension();
    const int kernelColCount = m_kernel.domainDimension();

    // Assert that the kernel tensor dimensions are compatible
    // with the number of components of the functions

    // TODO: This will need to be modified once we allow scalar-valued kernels
    // (treated as if they were multiplied by the unit tensor) with
    // vector-valued functions
    assert(testComponentCount == kernelRowCount);
    assert(kernelColCount == trialComponentCount);

    BasisData<ValueType> testBasisData, trialBasisData;
    GeometricalData<ValueType> testGeomData, trialGeomData;

    int testBasisDeps = 0, trialBasisDeps = 0;
    int testGeomDeps = INTEGRATION_ELEMENTS;
    int trialGeomDeps = INTEGRATION_ELEMENTS;

    m_testExpression.addDependencies(testBasisDeps, testGeomDeps);
    m_trialExpression.addDependencies(trialBasisDeps, trialGeomDeps);
    m_kernel.addGeometricalDependencies(testGeomDeps, trialGeomDeps);

    typedef typename GeometryFactory::Geometry Geometry;
    std::auto_ptr<Geometry> testGeometry(m_geometryFactory.make());
    std::auto_ptr<Geometry> trialGeometry(m_geometryFactory.make());

    arma::Cube<ValueType> testValues, trialValues;
    Array4D<ValueType> kernelValues(kernelRowCount, kernelColCount,
                                    testPointCount, trialPointCount);

    result.set_size(testDofCount, trialDofCount, geometryPairCount);

    testBasis.evaluate(testBasisDeps, m_localTestQuadPoints, ALL_DOFS, testBasisData);
    trialBasis.evaluate(trialBasisDeps, m_localTrialQuadPoints, ALL_DOFS, trialBasisData);

    // Iterate over the elements
    for (int pairIndex = 0; pairIndex < geometryPairCount; ++pairIndex)
    {
        setupGeometryConveniently(elementIndexPairs[pairIndex].first, *testGeometry);
        setupGeometryConveniently(elementIndexPairs[pairIndex].first, *trialGeometry);
        testGeometry->getData(testGeomDeps, m_localTestQuadPoints, testGeomData);
        trialGeometry->getData(trialGeomDeps, m_localTrialQuadPoints, trialGeomData);
        m_testExpression.evaluate(testBasisData, testGeomData, testValues);
        m_trialExpression.evaluate(trialBasisData, trialGeomData, trialValues);

        m_kernel.evaluateOnGrid(testGeomData, trialGeomData, kernelValues);

        // For now, we assume that the kernel is (general) tensorial,
        // later we might handle specially the case of it being a scalar
        // times the identity tensor.
        for (int trialDof = 0; trialDof < trialDofCount; ++trialDof)
            for (int testDof = 0; testDof < testDofCount; ++testDof)
            {
                ValueType sum = 0.;
                for (int trialPoint = 0; trialPoint < trialPointCount; ++trialPoint)
                    for (int testPoint = 0; testPoint < testPointCount; ++testPoint)
                        for (int trialDim = 0; trialDim < trialComponentCount; ++trialDim)
                            for (int testDim = 0; testDim < testComponentCount; ++testDim)
                                sum +=  m_testQuadWeights[testPoint] *
                                        testGeomData.integrationElements(testPoint) *
                                        testValues(testDim, testDof, testPoint) *
                                        kernelValues(testDim, trialDim, testPoint, trialPoint) *
                                        trialValues(trialDim, trialDof, trialPoint) *
                                        trialGeomData.integrationElements(trialPoint) *
                                        m_trialQuadWeights[trialPoint];
                result(testDof, trialDof, pairIndex) = sum;
            }
    }
}


template <typename ValueType, typename GeometryFactory>
void SeparableNumericalDoubleIntegrator<ValueType, GeometryFactory>::integrateCl(
	const OpenClFramework<ValueType,int> *openClFramework,
        CallVariant callVariant,
        const std::vector<int>& elementIndicesA,
        int elementIndexB,
        const Basis<ValueType>& basisA,
        const Basis<ValueType>& basisB,
        LocalDofIndex localDofIndexB,
        arma::Cube<ValueType>& result) const
{
    const int testPointCount = m_localTestQuadPoints.n_cols;
    const int trialPointCount = m_localTrialQuadPoints.n_cols;
    const int elementACount = elementIndicesA.size();
    const int pointDim = m_localTestQuadPoints.n_rows;
    const int meshDim = openClFramework->meshGeom().dim;

    if (testPointCount == 0 || trialPointCount == 0 || elementACount == 0)
        return;
    // TODO: in the (pathological) case that pointCount == 0 but
    // geometryCount != 0, set elements of result to 0.

    // Evaluate constants
    const int testComponentCount = m_testExpression.codomainDimension();
    const int trialComponentCount = m_trialExpression.codomainDimension();
    const int dofCountA = basisA.size();
    const int dofCountB = localDofIndexB == ALL_DOFS ? basisB.size() : 1;
    const int testDofCount = callVariant == TEST_TRIAL ? dofCountA : dofCountB;
    const int trialDofCount = callVariant == TEST_TRIAL ? dofCountB : dofCountA;

    const int kernelRowCount = m_kernel.codomainDimension();
    const int kernelColCount = m_kernel.domainDimension();

    // Assert that the kernel tensor dimensions are compatible
    // with the number of components of the functions

    // TODO: This will need to be modified once we allow scalar-valued kernels
    // (treated as if they were multiplied by the unit tensor) with
    // vector-valued functions
    assert(testComponentCount == kernelRowCount);
    assert(kernelColCount == trialComponentCount);

    int testBasisDeps = 0, trialBasisDeps = 0;
    int testGeomDeps = INTEGRATION_ELEMENTS;
    int trialGeomDeps = INTEGRATION_ELEMENTS;

    cl::Buffer *clElementIndicesA;
    cl::Buffer *clTrialQuadWeights;
    cl::Buffer *clTestQuadWeights;
    cl::Buffer *clGlobalTrialPoints;
    cl::Buffer *clGlobalTestPoints;
    cl::Buffer *clTrialValues;
    cl::Buffer *clTestValues;
    cl::Buffer *clTrialIntegrationElements;
    cl::Buffer *clTestIntegrationElements;
    cl::Buffer *clResult;

    m_testExpression.addDependencies(testBasisDeps, testGeomDeps);
    m_trialExpression.addDependencies(trialBasisDeps, trialGeomDeps);
    m_kernel.addGeometricalDependencies(testGeomDeps, trialGeomDeps);

    result.set_size(testDofCount, trialDofCount, elementACount);

    clElementIndicesA = openClFramework->pushIndexVector (elementIndicesA);
    clTrialQuadWeights = openClFramework->pushValueVector (m_trialQuadWeights);
    clTestQuadWeights = openClFramework->pushValueVector (m_testQuadWeights);
    clResult = openClFramework->createValueBuffer (testDofCount*trialDofCount*elementACount,
						   CL_MEM_WRITE_ONLY);


    // Build the OpenCL program
    std::vector<std::string> sources;
    sources.push_back (openClFramework->initStr());
    sources.push_back (basisA.clCodeString()); // note: need to collect separate string from basisB
    sources.push_back (m_kernel.evaluateClCode());
    sources.push_back (clStrIntegrateRowOrCol());
    openClFramework->loadProgramFromStringArray (sources);


    // Call the CL kernels to map the trial and test quadrature points
    if (callVariant == TEST_TRIAL)
    {
	clGlobalTestPoints = openClFramework->createValueBuffer(
            elementACount*testPointCount*meshDim, CL_MEM_READ_WRITE);
	clTestIntegrationElements = openClFramework->createValueBuffer(
	    elementACount*testPointCount, CL_MEM_READ_WRITE);
	cl::Kernel &clMapTest = openClFramework->setKernel ("clMapPointsToElements");
	clMapTest.setArg (0, openClFramework->meshGeom().cl_vtxbuf);
	clMapTest.setArg (1, openClFramework->meshGeom().nvtx);
	clMapTest.setArg (2, openClFramework->meshGeom().cl_elbuf);
	clMapTest.setArg (3, meshDim);
	clMapTest.setArg (4, openClFramework->meshGeom().nels);
	clMapTest.setArg (5, openClFramework->meshGeom().nidx);
	clMapTest.setArg (6, *clTestQuadPoints);
	clMapTest.setArg (7, testPointCount);
	clMapTest.setArg (8, pointDim);
	clMapTest.setArg (9, *clElementIndicesA);
	clMapTest.setArg (10, elementACount);
	clMapTest.setArg (11, *clGlobalTestPoints);
	clMapTest.setArg (12, *clTestIntegrationElements);
	openClFramework->enqueueKernel (cl::NDRange(elementACount, testPointCount));

        clGlobalTrialPoints = openClFramework->createValueBuffer (
	    trialPointCount*meshDim, CL_MEM_READ_WRITE);
	clTrialIntegrationElements = openClFramework->createValueBuffer(
	    trialPointCount, CL_MEM_READ_WRITE);
	cl::Kernel &clMapTrial = openClFramework->setKernel ("clMapPointsToElement");
	clMapTrial.setArg (0, openClFramework->meshGeom().cl_vtxbuf);
	clMapTrial.setArg (1, openClFramework->meshGeom().nvtx);
	clMapTrial.setArg (2, openClFramework->meshGeom().cl_elbuf);
	clMapTrial.setArg (3, meshDim);
	clMapTrial.setArg (4, openClFramework->meshGeom().nels);
	clMapTrial.setArg (5, openClFramework->meshGeom().nidx);
	clMapTrial.setArg (6, *clTestQuadPoints);
	clMapTrial.setArg (7, trialPointCount);
	clMapTrial.setArg (8, pointDim);
	clMapTrial.setArg (9, elementIndexB);
	clMapTrial.setArg (10, *clGlobalTrialPoints);
	clMapTrial.setArg (11, *clTrialIntegrationElements);
	openClFramework->enqueueKernel (cl::NDRange(trialPointCount));

	clTestValues = openClFramework->createValueBuffer (
	    elementACount*testPointCount*testDofCount, CL_MEM_READ_WRITE);
	cl::Kernel &clBasisTest = openClFramework->setKernel ("clBasisfElements");
	clBasisTest.setArg (0, openClFramework->meshGeom().cl_vtxbuf);
	clBasisTest.setArg (1, openClFramework->meshGeom().nvtx);
	clBasisTest.setArg (2, openClFramework->meshGeom().cl_elbuf);
	clBasisTest.setArg (3, meshDim);
	clBasisTest.setArg (4, openClFramework->meshGeom().nels);
	clBasisTest.setArg (5, openClFramework->meshGeom().nidx);
	clBasisTest.setArg (6, *clElementIndicesA);
	clBasisTest.setArg (7, elementACount);
	clBasisTest.setArg (8, *clTestQuadPoints);
	clBasisTest.setArg (9, testPointCount);
	clBasisTest.setArg (10, pointDim);
	clBasisTest.setArg (11, testDofCount);
	clBasisTest.setArg (12, *clTestValues);
	openClFramework->enqueueKernel (cl::NDRange(elementACount, testPointCount));

	clTrialValues = openClFramework->createValueBuffer (
	    trialPointCount*trialDofCount, CL_MEM_READ_WRITE);
	cl::Kernel &clBasisTrial = openClFramework->setKernel ("clBasisfElement");
	clBasisTrial.setArg (0, openClFramework->meshGeom().cl_vtxbuf);
	clBasisTrial.setArg (1, openClFramework->meshGeom().nvtx);
	clBasisTrial.setArg (2, openClFramework->meshGeom().cl_elbuf);
	clBasisTrial.setArg (3, meshDim);
	clBasisTrial.setArg (4, openClFramework->meshGeom().nels);
	clBasisTrial.setArg (5, openClFramework->meshGeom().nidx);
	clBasisTrial.setArg (6, elementIndexB);
	clBasisTrial.setArg (7, *clTrialQuadPoints);
	clBasisTrial.setArg (8, trialPointCount);
	clBasisTrial.setArg (9, pointDim);
	clBasisTrial.setArg (10, trialDofCount);
	clBasisTrial.setArg (11, localDofIndexB);
	clBasisTrial.setArg (12, *clTrialValues);
	openClFramework->enqueueKernel (cl::NDRange(trialPointCount));
    }
    else
    {
        clGlobalTrialPoints = openClFramework->createValueBuffer (
	    elementACount*trialPointCount*meshDim, CL_MEM_READ_WRITE);
	clTrialIntegrationElements = openClFramework->createValueBuffer(
	    elementACount*trialPointCount, CL_MEM_READ_WRITE);
	cl::Kernel &clMapTrial = openClFramework->setKernel ("clMapPointsToElements");
	clMapTrial.setArg (0, openClFramework->meshGeom().cl_vtxbuf);
	clMapTrial.setArg (1, openClFramework->meshGeom().nvtx);
	clMapTrial.setArg (2, openClFramework->meshGeom().cl_elbuf);
	clMapTrial.setArg (3, meshDim);
	clMapTrial.setArg (4, openClFramework->meshGeom().nels);
	clMapTrial.setArg (5, openClFramework->meshGeom().nidx);
	clMapTrial.setArg (6, *clTrialQuadPoints);
	clMapTrial.setArg (7, trialPointCount);
	clMapTrial.setArg (8, pointDim);
	clMapTrial.setArg (9, *clElementIndicesA);
	clMapTrial.setArg (10, elementACount);
	clMapTrial.setArg (11, *clGlobalTrialPoints);
	clMapTrial.setArg (12, *clTrialIntegrationElements);
	openClFramework->enqueueKernel (cl::NDRange(elementACount, trialPointCount));

	clGlobalTestPoints = openClFramework->createValueBuffer(
            testPointCount*meshDim, CL_MEM_READ_WRITE);
	clTestIntegrationElements = openClFramework->createValueBuffer(
	    testPointCount, CL_MEM_READ_WRITE);
	cl::Kernel &clMapTest = openClFramework->setKernel ("clMapPointsToElement");
	clMapTest.setArg (0, openClFramework->meshGeom().cl_vtxbuf);
	clMapTest.setArg (1, openClFramework->meshGeom().nvtx);
	clMapTest.setArg (2, openClFramework->meshGeom().cl_elbuf);
	clMapTest.setArg (3, meshDim);
	clMapTest.setArg (4, openClFramework->meshGeom().nels);
	clMapTest.setArg (5, openClFramework->meshGeom().nidx);
	clMapTest.setArg (6, *clTestQuadPoints);
	clMapTest.setArg (7, testPointCount);
	clMapTest.setArg (8, pointDim);
	clMapTest.setArg (9, elementIndexB);
	clMapTest.setArg (10, *clGlobalTestPoints);
	clMapTest.setArg (11, *clTestIntegrationElements);
	openClFramework->enqueueKernel (cl::NDRange(testPointCount));

	clTrialValues = openClFramework->createValueBuffer (
	    elementACount*trialPointCount*trialDofCount, CL_MEM_READ_WRITE);
	cl::Kernel &clBasisTrial = openClFramework->setKernel ("clBasisfElements");
	clBasisTrial.setArg (0, openClFramework->meshGeom().cl_vtxbuf);
	clBasisTrial.setArg (1, openClFramework->meshGeom().nvtx);
	clBasisTrial.setArg (2, openClFramework->meshGeom().cl_elbuf);
	clBasisTrial.setArg (3, meshDim);
	clBasisTrial.setArg (4, openClFramework->meshGeom().nels);
	clBasisTrial.setArg (5, openClFramework->meshGeom().nidx);
	clBasisTrial.setArg (6, *clElementIndicesA);
	clBasisTrial.setArg (7, elementACount);
	clBasisTrial.setArg (8, *clTrialQuadPoints);
	clBasisTrial.setArg (9, trialPointCount);
	clBasisTrial.setArg (10, pointDim);
	clBasisTrial.setArg (11, trialDofCount);
	clBasisTrial.setArg (12, *clTrialValues);
	openClFramework->enqueueKernel (cl::NDRange(elementACount, trialPointCount));

	clTestValues = openClFramework->createValueBuffer (
	    testPointCount*testDofCount, CL_MEM_READ_WRITE);
	cl::Kernel &clBasisTest = openClFramework->setKernel ("clBasisfElement");
	clBasisTest.setArg (0, openClFramework->meshGeom().cl_vtxbuf);
	clBasisTest.setArg (1, openClFramework->meshGeom().nvtx);
	clBasisTest.setArg (2, openClFramework->meshGeom().cl_elbuf);
	clBasisTest.setArg (3, meshDim);
	clBasisTest.setArg (4, openClFramework->meshGeom().nels);
	clBasisTest.setArg (5, openClFramework->meshGeom().nidx);
	clBasisTest.setArg (6, elementIndexB);
	clBasisTest.setArg (7, *clTestQuadPoints);
	clBasisTest.setArg (8, testPointCount);
	clBasisTest.setArg (9, pointDim);
	clBasisTest.setArg (10, testDofCount);
	clBasisTest.setArg (11, localDofIndexB);
	clBasisTest.setArg (12, *clTestValues);
	openClFramework->enqueueKernel (cl::NDRange(testPointCount));
    }



    // Build the OpenCL kernel
    cl::Kernel &clKernel = openClFramework->setKernel ("clIntegrate");

    // Set kernel arguments
    int argIdx = 0;
    clKernel.setArg (argIdx++, openClFramework->meshGeom().cl_vtxbuf);
    clKernel.setArg (argIdx++, openClFramework->meshGeom().nvtx);
    clKernel.setArg (argIdx++, openClFramework->meshGeom().cl_elbuf);
    clKernel.setArg (argIdx++, openClFramework->meshGeom().dim);
    clKernel.setArg (argIdx++, openClFramework->meshGeom().nels);
    clKernel.setArg (argIdx++, openClFramework->meshGeom().nidx);
    clKernel.setArg (argIdx++, *clGlobalTrialPoints);
    clKernel.setArg (argIdx++, *clGlobalTestPoints);
    clKernel.setArg (argIdx++, *clTrialIntegrationElements);
    clKernel.setArg (argIdx++, *clTestIntegrationElements);
    clKernel.setArg (argIdx++, *clTrialValues);
    clKernel.setArg (argIdx++, *clTestValues);
    clKernel.setArg (argIdx++, *clTrialQuadWeights);
    clKernel.setArg (argIdx++, *clTestQuadWeights);
    clKernel.setArg (argIdx++, trialPointCount);
    clKernel.setArg (argIdx++, testPointCount);
    clKernel.setArg (argIdx++, trialComponentCount);
    clKernel.setArg (argIdx++, testComponentCount);
    clKernel.setArg (argIdx++, trialDofCount);
    clKernel.setArg (argIdx++, testDofCount);
    clKernel.setArg (argIdx++, elementACount);
    clKernel.setArg (argIdx++, callVariant == TEST_TRIAL ? 1:0);
    clKernel.setArg (argIdx++, *clElementIndicesA);
    clKernel.setArg (argIdx++, elementIndexB);
    clKernel.setArg (argIdx++, *clResult);


    // Run the CL kernel
    openClFramework->enqueueKernel (cl::NDRange(elementACount));

    // Copy results back
    openClFramework->pullValueCube (*clResult, result);
    
    // Clean up local device buffers
    delete clElementIndicesA;
    delete clTrialQuadWeights;
    delete clTestQuadWeights;
    delete clGlobalTrialPoints;
    delete clGlobalTestPoints;
    delete clTestValues;
    delete clTrialValues;
    delete clTrialIntegrationElements;
    delete clTestIntegrationElements;
    delete clResult;
}

template <typename ValueType, typename GeometryFactory>
const std::string SeparableNumericalDoubleIntegrator<ValueType, GeometryFactory>::clStrIntegrateRowOrCol () const
{
  return std::string (separable_numerical_double_integrator_cl,
		      separable_numerical_double_integrator_cl_len);
}


} // namespace Fiber
