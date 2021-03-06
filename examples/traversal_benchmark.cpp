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

#include <iostream>
#include <memory> // auto_ptr
#include <sys/time.h>

#include "grid/dune.hpp"
#include "grid/entity.hpp"
#include "grid/entity_iterator.hpp"
#include "grid/geometry.hpp"
#include "grid/grid.hpp"
#include "grid/grid_factory.hpp"
#include "grid/grid_view.hpp"
#include "grid/structured_grid_factory.hpp"

using namespace Bempp;

/** Create a structured triangular grid of 2 * N_ELEMENTS * (N_ELEMENTS + 1)
elements and iterate over entities. If CALC_CENTER is set, for each entity
calculate additionally its center. */

int main()
{
    // Benchmark parameters
    const int N_ELEMENTS = 100;
    const int N_TRIALS = 20;
    const bool CALC_CENTER = true;

    if (CALC_CENTER)
        std::cout << "Benchmark variant: full loops\n";
    else
        std::cout << "Benchmark variant: empty loops\n";

    //////////////////// BEMPP OBJECTS ///////////////////

    std::cout << "\nUsing Bempp objects..." << std::endl;

    // Create a structured grid
    GridParameters params;
    params.topology = GridParameters::TRIANGULAR;

    const int dimGrid = 2;
    arma::Col<ctype> lowerLeft(dimGrid);
    arma::Col<ctype> upperRight(dimGrid);
    arma::Col<unsigned int> nElements(dimGrid);
    lowerLeft.fill(0);
    upperRight.fill(1);
    nElements(0) = N_ELEMENTS;
    nElements(1) = N_ELEMENTS + 1;

    std::auto_ptr<Grid> grid(GridFactory::createStructuredGrid(params, lowerLeft, upperRight, nElements));
    std::cout << nElements[0] * nElements[1] << " elements created\n";

    // Create a leaf view
    std::auto_ptr<GridView> leafGridView(grid->leafView());

    {
        std::cout << "Iterating over faces..." << std::endl;

        volatile int j = 0;
        timeval start, end;
        gettimeofday(&start, 0);

        for (int i = 0; i < N_TRIALS; ++i) {
            std::auto_ptr<EntityIterator<0> > leafFaceIt = leafGridView->entityIterator<0>();
            while (!leafFaceIt->finished()) {
                ++j;
                if (CALC_CENTER) {
                    const Entity<0>& e = leafFaceIt->entity();
                    Dune::GeometryType gt = e.type();
                    const Geometry& geo = e.geometry();
                    arma::Col<double> elementCenter;
                    geo.center(elementCenter);
                }
                leafFaceIt->next();
            }
        }
        gettimeofday(&end, 0);

        double total_time = (end.tv_sec + end.tv_usec / 1000000.) -
                            (start.tv_sec + start.tv_usec / 1000000.);
        std::cout << "Traversed faces: " << j << '\n';
        std::cout << "Traversal time: " << total_time << std::endl;
    }

    {
        std::cout << "Iterating over vertices..." << std::endl;

        volatile int j = 0;
        timeval start, end;
        gettimeofday(&start, 0);

        for (int i = 0; i < N_TRIALS; ++i) {
            std::auto_ptr<EntityIterator<2> > leafVertexIt = leafGridView->entityIterator<2>();
            while (!leafVertexIt->finished()) {
                ++j;
                if (CALC_CENTER) {
                    const Entity<2>& e = leafVertexIt->entity();
                    GeometryType gt = e.type();
                    const Geometry& geo = e.geometry();
                    arma::Col<double> elementCenter;
                    geo.center(elementCenter);
                }
                leafVertexIt->next();
            }
        }
        gettimeofday(&end, 0);

        double total_time = (end.tv_sec + end.tv_usec / 1000000.) -
                            (start.tv_sec + start.tv_usec / 1000000.);
        std::cout << "Traversed vertices: " << j << '\n';
        std::cout << "Traversal time: " << total_time << std::endl;
    }

    //////////////////// DUNE OBJECTS ///////////////////

    std::cout << "\nUsing Dune objects..." << std::endl;

    Dune::FieldVector<ctype,dimGrid> duneLowerLeft;
    duneLowerLeft[0] = duneLowerLeft[1] = 0;
    Dune::FieldVector<ctype,dimGrid> duneUpperRight;
    duneUpperRight[0] = duneUpperRight[1] = 1;
    Dune::array<unsigned int,dimGrid> duneNElements;
    duneNElements[0] = N_ELEMENTS;
    duneNElements[1] = N_ELEMENTS + 1;

    std::auto_ptr<DefaultDuneGrid> duneGrid =
        Dune::BemppStructuredGridFactory<DefaultDuneGrid>::
        createSimplexGrid(duneLowerLeft, duneUpperRight, duneNElements);
    std::cout << nElements[0] * nElements[1] << " elements created\n";

    const int dimWorld = 3;

    DefaultDuneGrid::LeafGridView duneLeafGridView = duneGrid->leafView();

    {
        std::cout << "Iterating over faces..." << std::endl;

        volatile int j = 0;
        timeval start, end;
        gettimeofday(&start, 0);

        for (int i = 0; i < N_TRIALS; ++i) {
            typedef DefaultDuneGrid::LeafGridView::Codim<0> Codim;
            Codim::Iterator leafFaceIt = duneLeafGridView.begin<0>();
            Codim::Iterator leafEnd = duneLeafGridView.end<0>();
            for (; leafFaceIt != leafEnd; ++leafFaceIt) {
                ++j;
                if (CALC_CENTER) {
                    const Codim::Entity& e = *leafFaceIt;
                    Dune::GeometryType gt = e.type();
                    const Codim::Entity::Geometry& geo = e.geometry();
                    Dune::FieldVector<ctype, dimWorld> elementCenter = geo.center();
                }
            }
        }
        gettimeofday(&end, 0);

        double total_time = (end.tv_sec + end.tv_usec / 1000000.) -
                            (start.tv_sec + start.tv_usec / 1000000.);
        std::cout << "Traversed faces: " << j << '\n';
        std::cout << "Traversal time: " << total_time << std::endl;
    }


    {
        std::cout << "Iterating over vertices..." << std::endl;

        volatile int j = 0;
        timeval start, end;
        gettimeofday(&start, 0);

        for (int i = 0; i < N_TRIALS; ++i) {
            typedef DefaultDuneGrid::LeafGridView::Codim<2> Codim;
            Codim::Iterator leafVertexIt = duneLeafGridView.begin<2>();
            Codim::Iterator leafEnd = duneLeafGridView.end<2>();
            for (; leafVertexIt != leafEnd; ++leafVertexIt) {
                ++j;
                if (CALC_CENTER) {
                    const Codim::Entity& e = *leafVertexIt;
                    Dune::GeometryType gt = e.type();
                    const Codim::Entity::Geometry& geo = e.geometry();
                    Dune::FieldVector<ctype, dimWorld> elementCenter = geo.center();
                }
            }
        }
        gettimeofday(&end, 0);

        double total_time = (end.tv_sec + end.tv_usec / 1000000.) -
                            (start.tv_sec + start.tv_usec / 1000000.);
        std::cout << "Traversed vertices: " << j << '\n';
        std::cout << "Traversal time: " << total_time << std::endl;
    }

}
