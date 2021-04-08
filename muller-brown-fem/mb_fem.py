from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Make domain specific to problem
react_min = np.genfromtxt('react_min.txt')
prod_min = np.genfromtxt('prod_min.txt')
print(react_min)
print(prod_min)
radii = 0.1
rectangle = Rectangle(Point(-1.5,-0.5), Point(1.0,2.0))
react_domain = Circle(Point(react_min[0],react_min[1]), radii) 
prod_domain = Circle(Point(prod_min[0],prod_min[1]), radii) 
domain = rectangle

# Make subdomains
domain.set_subdomain(1, react_domain)
domain.set_subdomain(2, prod_domain)

# Generate mesh
mesh = generate_mesh(domain, 200)

# Create boundaries
boundary_markers = MeshFunction("size_t", mesh, 2, mesh.domains())
boundaries_react = MeshFunction("size_t", mesh, 1, mesh.domains())
boundaries_prod = MeshFunction("size_t", mesh, 1, mesh.domains())

# Use the cell domains to set the boundaries
for f in facets(mesh):
    domains = []
    for c in cells(f):
        domains.append(boundary_markers[c])
    domains = list(set(domains))
    #if len(domains) > 1:
    for i in domains:
        if i == 1:
            boundaries_react[f] = 2
        elif i == 2:
            boundaries_prod[f] = 2

# Make function space
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
bc_react = DirichletBC(V, Constant(0), boundaries_react, 2)
bc_prod = DirichletBC(V, Constant(1), boundaries_prod, 2)
bcs = [bc_react, bc_prod]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f_0 = Expression('A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-20, a_=-1, b_=0, c_=-10, x_=1, y_=0)
f_0_x = Expression('A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-20, a_=-1, b_=0, c_=-10, x_=1, y_=0)
f_0_y = Expression('A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-20, a_=-1, b_=0, c_=-10, x_=1, y_=0)
f_0_grad = as_vector((f_0_x,f_0_y))
f_1 = Expression('A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-10, a_=-1, b_=0, c_=-10, x_=0, y_=0.5)
f_1_x = Expression('A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-10, a_=-1, b_=0, c_=-10, x_=0, y_=0.5)
f_1_y = Expression('A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-10, a_=-1, b_=0, c_=-10, x_=0, y_=0.5)
f_1_grad = as_vector((f_1_x,f_1_y))
f_2 = Expression('A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-17, a_=-6.5, b_=11, c_=-6.5, x_=-0.5, y_=1.5)
f_2_x = Expression('A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-17, a_=-6.5, b_=11, c_=-6.5, x_=-0.5, y_=1.5)
f_2_y = Expression('A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-17, a_=-6.5, b_=11, c_=-6.5, x_=-0.5, y_=1.5)
f_2_grad = as_vector((f_2_x,f_2_y))
f_3 = Expression('A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=1.5, a_=0.7, b_=0.6, c_=0.7, x_=-1, y_=1)
f_3_x = Expression('A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=1.5, a_=0.7, b_=0.6, c_=0.7, x_=-1, y_=1)
f_3_y = Expression('A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=1.5, a_=0.7, b_=0.6, c_=0.7, x_=-1, y_=1)
f_3_grad = as_vector((f_3_x,f_3_y))
f_total = f_0+f_1+f_2+f_3
f_grad_total = f_0_grad+f_1_grad+f_2_grad+f_3_grad
#f_total = Expression('10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)', degree=2)
beta = Constant('1')
a = (dot(grad(u), grad(v))+beta*dot(f_grad_total,grad(u))*v)*dx
L = Constant('0')*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Plot solution and mesh
plot(u)
#plot(mesh)
plt.show()

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u
