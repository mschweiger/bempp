// Copyright (C) 2011 by the BEM++ Authors
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

#ifndef bempp_concrete_geometry_hpp
#define bempp_concrete_geometry_hpp

#include "geometry.hpp"
#include "geometry_type.hpp"
#include "common.hpp"

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <armadillo>

namespace Bempp
{

/** \brief Wrapper of a Dune geometry of type \p DuneGeometry */

template<typename DuneGeometry>
class ConcreteGeometry : public Geometry
{
private:
    const DuneGeometry* m_dune_geometry;

    /** \brief Default constructor.

    \internal Should be used only by friend classes that call setDuneGeometry() later on. */
    ConcreteGeometry() : m_dune_geometry(0) {
    }

    void setDuneGeometry(const DuneGeometry* dune_geometry) {
        m_dune_geometry = dune_geometry;
    }

    template<int codim, typename DuneEntity> friend class ConcreteEntity;

public:
    /** \brief Constructor from a pointer to DuneGeometry.

      This object does not assume ownership of \p *dune_geometry.
    */
    explicit ConcreteGeometry(const DuneGeometry* dune_geometry) :
        m_dune_geometry(dune_geometry) {}

    /** \brief Read-only access to the underlying Dune geometry object. */
    const DuneGeometry& duneGeometry() const {
        return *m_dune_geometry;
    }

    virtual GeometryType type() const {
        return m_dune_geometry->type();
    }

    virtual bool affine() const {
        return m_dune_geometry->affine();
    }

    virtual int cornerCount() const {
        return m_dune_geometry->corners();
    }

    virtual void corners(arma::Mat<ctype>& c) const {
        const int cdim = DuneGeometry::dimensionworld;
        const int n = m_dune_geometry->corners();
        c.set_size(cdim, n);

        /** \fixme In future this copying should be optimised away by casting
        appropriate columns of c to Dune field vectors. But this
        can't be done until unit tests are in place. */
        typename DuneGeometry::GlobalCoordinate g;
        for (int j = 0; j < n; ++j) {
            g = m_dune_geometry->corner(j);
            for (int i = 0; i < cdim; ++i)
                c(i,j) = g[i];
        }
    }

    virtual void local2global(const arma::Mat<ctype>& local,
                              arma::Mat<ctype>& global) const {
        const int mdim = DuneGeometry::mydimension;
        const int cdim = DuneGeometry::coorddimension;
#ifndef NDEBUG
        if (local.n_rows != mdim)
            throw std::invalid_argument("Geometry::local2global(): invalid dimensions of the 'local' array");
#endif
        const int n = local.n_cols;
        global.set_size(cdim, n);

        /** \fixme Optimise (get rid of data copying). */
        typename DuneGeometry::GlobalCoordinate g;
        typename DuneGeometry::LocalCoordinate l;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < mdim; ++i)
                l[i] = local(i,j);
            g = m_dune_geometry->global(l);
            for (int i = 0; i < cdim; ++i)
                global(i,j) = g[i];
        }
    }

    virtual void global2local(const arma::Mat<ctype>& global,
                              arma::Mat<ctype>& local) const {
        const int mdim = DuneGeometry::mydimension;
        const int cdim = DuneGeometry::coorddimension;
#ifndef NDEBUG
        if (global.n_rows != cdim)
            throw std::invalid_argument("Geometry::global2local(): invalid dimensions of the 'global' array");
#endif
        const int n = global.n_cols;
        local.set_size(mdim, n);

        /** \fixme Optimise (get rid of data copying). */
        typename DuneGeometry::GlobalCoordinate g;
        typename DuneGeometry::LocalCoordinate l;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < cdim; ++i)
                g[i] = global(i,j);
            l = m_dune_geometry->local(g);
            for (int i = 0; i < mdim; ++i)
                local(i,j) = l[i];
        }
    }

    virtual void integrationElement(const arma::Mat<ctype>& local,
                                    arma::Row<ctype>& int_element) const {
        const int mdim = DuneGeometry::mydimension;
#ifndef NDEBUG
        if (local.n_rows != mdim)
            throw std::invalid_argument("Geometry::local2global(): invalid dimensions of the 'local' array");
#endif
        const int n = local.n_cols;
        int_element.set_size(n);

        /** \fixme Optimise (get rid of data copying). */
        typename DuneGeometry::LocalCoordinate l;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < mdim; ++i)
                l[i] = local(i,j);
            ctype ie = m_dune_geometry->integrationElement(l);
            int_element(j) = ie;
        }
    }

    virtual ctype volume() const {
        return m_dune_geometry->volume();
    }

    virtual void center(arma::Col<ctype>& c) const {
        const int cdim = DuneGeometry::coorddimension;
        c.set_size(cdim);

        /** \fixme Optimise (get rid of data copying). */
        typename DuneGeometry::GlobalCoordinate g = m_dune_geometry->center();
        for (int i = 0; i < cdim; ++i)
            c(i) = g[i];
    }

    virtual void jacobianTransposed(const arma::Mat<ctype>& local,
                                    arma::Cube<ctype>& jacobian_t) const {
        const int mdim = DuneGeometry::mydimension;
        const int cdim = DuneGeometry::coorddimension;
#ifndef NDEBUG
        if (local.n_rows != mdim)
            throw std::invalid_argument("Geometry::jacobianTransposed(): invalid dimensions of the 'local' array");
#endif
        const int n = local.n_cols;
        jacobian_t.set_size(mdim, cdim, n);

        /** \bug Unfortunately Dune::FieldMatrix (the underlying type of
        JacobianTransposed) stores elements rowwise, while Armadillo does it
        columnwise. Hence element-by-element filling of jacobian_t seems
        unavoidable). */
        typename DuneGeometry::JacobianTransposed j_t;
        typename DuneGeometry::LocalCoordinate l;
        for (int k = 0; k < n; ++k) {
            /** \fixme However, this bit of data copying could be avoided. */
            for (int i = 0; i < mdim; ++i)
                l[i] = local(i,k);
            j_t = m_dune_geometry->jacobianTransposed(l);
            for (int j = 0; j < cdim; ++j)
                for (int i = 0; i < mdim; ++i)
                    jacobian_t(i,j,k) = j_t[i][j];
        }
    }

    virtual void jacobianInverseTransposed(const arma::Mat<ctype>& local,
                                           arma::Cube<ctype>& jacobian_inv_t) const {
        const int mdim = DuneGeometry::mydimension;
        const int cdim = DuneGeometry::coorddimension;
#ifndef NDEBUG
        if (local.n_rows != mdim)
            throw std::invalid_argument("Geometry::jacobianInverseTransposed(): invalid dimensions of the 'local' array");
#endif
        const int n = local.n_cols;
        jacobian_inv_t.set_size(cdim, mdim, n);

        /** \bug Unfortunately Dune::FieldMatrix (the underlying type of
        Jacobian) stores elements rowwise, while Armadillo does it
        columnwise. Hence element-by-element filling of jacobian_t seems
        unavoidable). */
        typename DuneGeometry::Jacobian j_inv_t;
        typename DuneGeometry::LocalCoordinate l;
        for (int k = 0; k < n; ++k) {
            /** \fixme However, this bit of data copying could be avoided. */
            for (int i = 0; i < mdim; ++i)
                l[i] = local(i,k);
            j_inv_t = m_dune_geometry->jacobianInverseTransposed(l);
            for (int j = 0; j < mdim; ++j)
                for (int i = 0; i < cdim; ++i)
                    jacobian_inv_t(i,j,k) = j_inv_t[i][j];
        }
    }
};

} // namespace Bempp

#endif
