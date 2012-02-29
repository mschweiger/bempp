// Copyright (C) 2011-2012 by the Fiber Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef fiber_separable_numerical_double_integrator_hpp
#define fiber_separable_numerical_double_integrator_hpp

#include "double_integrator.hpp"
#include "opencl_options.hpp"
#include "opencl_framework.hpp"

namespace Fiber
{

template <typename ValueType> class Expression;
template <typename ValueType> class Kernel;

/** \brief Integration over pairs of elements. */
template <typename ValueType, typename GeometryFactory>
class SeparableNumericalDoubleIntegrator :
        public DoubleIntegrator<ValueType>
{
public:
    typedef typename DoubleIntegrator<ValueType>::ElementIndexPair
    ElementIndexPair;

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
	    const OpenClFramework<ValueType,int> &openClFramework,
            const OpenClOptions& openClOptions);

    virtual void integrate(
            CallVariant callVariant,
            const std::vector<int>& elementIndicesA,
            int elementIndexB,
            const Basis<ValueType>& basisA,
            const Basis<ValueType>& basisB,
            LocalDofIndex localDofIndexB,
            arma::Cube<ValueType>& result) const;

    virtual void integrate(
            const std::vector<ElementIndexPair>& elementIndexPairs,
            const Basis<ValueType>& testBasis,
            const Basis<ValueType>& trialBasis,
            arma::Cube<ValueType>& result) const;

private:
    void setupGeometry(int elementIndex,
                       typename GeometryFactory::Geometry& geometry) const;

private:
    arma::Mat<ValueType> m_localTestQuadPoints;
    arma::Mat<ValueType> m_localTrialQuadPoints;
    std::vector<ValueType> m_testQuadWeights;
    std::vector<ValueType> m_trialQuadWeights;

    const GeometryFactory& m_geometryFactory;
    const arma::Mat<ValueType>& m_vertices;
    const arma::Mat<int>& m_elementCornerIndices;
    const arma::Mat<char>& m_auxElementData;

    const Expression<ValueType>& m_testExpression;
    const Kernel<ValueType>& m_kernel;
    const Expression<ValueType>& m_trialExpression;
    const OpenClFramework<ValueType,int>& m_openClFramework;
    OpenClOptions m_openClOptions;    
};

struct Element2D
{
    int vertexIndices[4];
    // potentially additional geometrical information
};

template <typename ValueType>
struct Vertex3D
{
    ValueType x, y, z;
};



} // namespace Fiber

#include "separable_numerical_double_integrator_imp.hpp"

#endif
