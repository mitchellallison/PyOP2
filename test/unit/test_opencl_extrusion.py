import os
import re

from firedrake import *

from pyop2 import op2

import pytest

import numpy

backends = ['opencl', 'sequential']

discretisations = (('CG', 1), ('CG', 2), ('DG', 0), ('DG', 1), ('DG', 2))


def setup_module(module):
    directory = os.path.join(os.path.dirname(__file__), '../data/')
    if not os.path.exists(directory):
        os.makedirs(directory)


def compare_results(expected, actual, epsilon):
    delta = expected - actual
    max_delta = numpy.max(numpy.abs(delta))
    assert max_delta <= epsilon


@pytest.fixture(scope='function', params=[(i, layers) for i in [1, 10, 100] for layers in [10, 100]],
                ids=["{}x{}-{}".format(i, i, layers) for i in [1, 10, 100] for layers in [10, 100]])
def mesh(request):
    (i, layers) = request.param
    mesh = UnitSquareMesh(i, i)
    mesh = ExtrudedMesh(mesh, layers=layers, layer_height=0.1)
    return mesh


@pytest.fixture(scope='function')
def test_name(request):
    return re.sub("(" + "|".join(backends) + ")-", "", request.node.name)


class TestOpenCLExtrusion:

    """
    OpenCL Extruded Mesh Tests
    """

    def test_extruded_simple_kernel(self, backend, discretisation, mesh, test_name, generate_extr_data):
        ((fam, deg), (vfam, vdeg)) = discretisation

        V = FunctionSpace(mesh, fam, deg, vfamily=vfam, vdegree=vdeg)
        name = "%s%dx%s%d" % (fam, deg, vfam, vdeg)
        f = Function(V)
        f.assign(10.0)

        k = op2.Kernel('''
        static inline void %(name)s(double* x[%(dim)d]) {
            for ( int i = 0; i < %(dim)d; i++ ) *x[i] += 1.0;
        }''' % {'name': name,
                'dim': V.cell_node_map().arity}, name=name)
        op2.par_loop(k, mesh.cell_set,
                     f.dat(op2.INC, V.cell_node_map()))

        file_name = os.path.join(os.path.dirname(__file__), '../data/{}.npy'.format(test_name))
        if generate_extr_data:
            numpy.save(file_name, f.dat.data)
        else:
            compare_results(numpy.load(file_name), f.dat.data, 0)

    def test_extruded_simple_kernel_coords(self, backend, discretisation, mesh, test_name, generate_extr_data):
        ((fam, deg), (vfam, vdeg)) = discretisation

        name = "%s%dx%s%d" % (fam, deg, vfam, vdeg)

        k = op2.Kernel('''
        static inline void %(name)s(double** x) {
            for ( int i = 0; i < %(dim)d; i++ ) {
                x[i][0] += 1.0;
                x[i][1] += 1.0;
                x[i][2] += 1.0;
            }
        }''' % {'name': name,
                'dim': mesh.coordinates.cell_node_map().arity}, name=name)

        op2.par_loop(k, mesh.cell_set,
                     mesh.coordinates.dat(op2.INC, mesh.coordinates.cell_node_map()))

        file_name = os.path.join(os.path.dirname(__file__), '../data/{}.npy'.format(test_name))
        if generate_extr_data:
            numpy.save(file_name, mesh.coordinates.dat.data)
        else:
            compare_results(numpy.load(file_name), mesh.coordinates.dat.data, 0)

    def test_extruded_simple_kernel_vector_function_spaces(self, backend, discretisation, mesh, test_name, generate_extr_data):
        ((fam, deg), (vfam, vdeg)) = discretisation

        V = VectorFunctionSpace(mesh, fam, deg, vfamily=vfam, vdegree=vdeg, dim=3)
        name = "%s%dx%s%d" % (fam, deg, vfam, vdeg)
        f = Function(V)
        f.assign(10.0)

        k = op2.Kernel('''
        static inline void %(name)s(double** x) {
            for ( int i = 0; i < %(dim)d; i++ ) {
                x[i][0] += 1.0;
                x[i][1] += 1.0;
                x[i][2] += 1.0;
            }
        }''' % {'name': name,
                'dim': V.cell_node_map().arity}, name=name)

        op2.par_loop(k, mesh.cell_set,
                     f.dat(op2.INC, V.cell_node_map()))

        file_name = os.path.join(os.path.dirname(__file__), '../data/{}.npy'.format(test_name))
        if generate_extr_data:
            numpy.save(file_name, f.dat.data)
        else:
            compare_results(numpy.load(file_name), f.dat.data, 0)

    def test_extruded_simple_kernel_rhs_assembly(self, backend, discretisation, mesh, test_name, generate_extr_data):
        ((fam, deg), (vfam, vdeg)) = discretisation

        V = FunctionSpace(mesh, fam, deg, vfamily=vfam, vdegree=vdeg)
        v = TestFunction(V)
        rhs = v * dx
        f = assemble(rhs)

        file_name = os.path.join(os.path.dirname(__file__), '../data/{}.npy'.format(test_name))
        if generate_extr_data:
            numpy.save(file_name, f.dat.data)
        else:
            compare_results(numpy.load(file_name), f.dat.data, 1e-17)
