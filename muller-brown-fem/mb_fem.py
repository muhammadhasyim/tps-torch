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
radii = 0.025
rectangle = Rectangle(Point(-1.5,-0.5), Point(1.0,2.0))
react_domain = Circle(Point(react_min[0],react_min[1]), radii) 
prod_domain = Circle(Point(prod_min[0],prod_min[1]), radii) 
domain = rectangle

# Make subdomains
domain.set_subdomain(1, react_domain)
domain.set_subdomain(2, prod_domain)

# Generate mesh
mesh = generate_mesh(domain, 50)

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
beta = Constant('1')
f_0 = Expression('A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-20, a_=-1, b_=0, c_=-10, x_=1, y_=0)
f_0_exp = Expression('exp(-beta*A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_)))', degree=2, A_=-20, a_=-1, b_=0, c_=-10, x_=1, y_=0, beta=1)
f_0_x = Expression('A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-20, a_=-1, b_=0, c_=-10, x_=1, y_=0)
f_0_y = Expression('A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-20, a_=-1, b_=0, c_=-10, x_=1, y_=0)
f_0_grad = as_vector((f_0_x,f_0_y))
f_1 = Expression('A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-10, a_=-1, b_=0, c_=-10, x_=0, y_=0.5)
f_1_exp = Expression('exp(-beta*A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_)))', degree=2, A_=-10, a_=-1, b_=0, c_=-10, x_=0, y_=0.5, beta=1)
f_1_x = Expression('A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-10, a_=-1, b_=0, c_=-10, x_=0, y_=0.5)
f_1_y = Expression('A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-10, a_=-1, b_=0, c_=-10, x_=0, y_=0.5)
f_1_grad = as_vector((f_1_x,f_1_y))
f_2 = Expression('A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-17, a_=-6.5, b_=11, c_=-6.5, x_=-0.5, y_=1.5)
f_2_exp = Expression('exp(-beta*A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_)))', degree=2, A_=-17, a_=-6.5, b_=11, c_=-6.5, x_=-0.5, y_=1.5, beta=1)
f_2_x = Expression('A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-17, a_=-6.5, b_=11, c_=-6.5, x_=-0.5, y_=1.5)
f_2_y = Expression('A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=-17, a_=-6.5, b_=11, c_=-6.5, x_=-0.5, y_=1.5)
f_2_grad = as_vector((f_2_x,f_2_y))
f_3 = Expression('A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=1.5, a_=0.7, b_=0.6, c_=0.7, x_=-1, y_=1)
f_3_exp = Expression('exp(-beta*A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_)))', degree=2, A_=1.5, a_=0.7, b_=0.6, c_=0.7, x_=-1, y_=1, beta=1)
f_3_x = Expression('A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=1.5, a_=0.7, b_=0.6, c_=0.7, x_=-1, y_=1)
f_3_y = Expression('A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))', degree=2, A_=1.5, a_=0.7, b_=0.6, c_=0.7, x_=-1, y_=1)
f_3_grad = as_vector((f_3_x,f_3_y))
f_total = f_0+f_1+f_2+f_3
f_total_exp = f_0_exp*f_1_exp*f_2_exp*f_3_exp
f_grad_total = f_0_grad+f_1_grad+f_2_grad+f_3_grad
#f_total = Expression('10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)', degree=2)
beta = Constant('1')
a = (dot(grad(u), grad(v))+beta*dot(f_grad_total,grad(u))*v)*dx
L = Constant('0')*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs, solver_parameters={'linear_solver' : 'mumps'})

vertex_values = u.compute_vertex_values(mesh)
coordinates = mesh.coordinates()
np.savetxt("vertex_values.txt", vertex_values)
np.savetxt("vertex_coords.txt", coordinates)

# Evaluate u at points
#p = Point(-0.5,1.5)
p = Point(-0.6,0.5)
# print(u(p))
# p = Point(react_min[0],react_min[1])
# print(u(p))
# p = Point(prod_min[0],prod_min[1])
# print(u(p))
# Make vector function space
# V_vec = VectorFunctionSpace(mesh, 'P', 1)
# u_grad = project(grad(u),V_vec)
gu = grad(u)
Gu = project(gu, solver_type="cg")
#Gu = dot(Gu,Gu)
cost = assemble(dot(grad(u), grad(u))*f_total_exp*dx)
print(cost)
#print(Gu(p))
#test = Gu(p)
#print(test[0]*test[0]+test[1]*test[1])
#print(u_grad(p))
#u_grad_2 = dot(grad(u), grad(u))
#print(u_grad_2(p))

# Plot solution and mesh
# plot(u)
#plot(mesh)
#plt.show()

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# Evaluate points on structured grid
n_struct = 100
points_x = np.linspace(-1.5,1.0,n_struct)
points_y = np.linspace(-0.5,2.0,n_struct)
xx, yy = np.meshgrid(points_x,points_y)
zz = np.zeros_like(xx)
grad_2_zz = np.zeros_like(xx)
for i in range(n_struct):
    for j in range(n_struct):
        p = Point(xx[i][j],yy[i][j])
        zz[i][j] = u(p)
        test = Gu(p)
        grad_2_zz[i][j] = test[0]*test[0]+test[1]*test[1]

np.savetxt("vertex_values_struct.txt", vertex_values)
np.savetxt("vertex_coords_struct.txt", coordinates)
with open('vertex_values_struct.txt', 'w') as outf:
    for i in range(n_struct):
        for j in range(n_struct):
            outf.write("{:.6g}\n".format(zz[i][j]))

with open('vertex_coords_struct.txt', 'w') as outf:
    for i in range(n_struct):
        for j in range(n_struct):
            outf.write("{:.6g} {:.6g}\n".format(xx[i][j],yy[i][j]))

A = np.array([-20,-10,-17,1.5])
a = np.array([-1,-1,-6.5,0.7])
b = np.array([0,0,11,0.6])
c = np.array([-10,-10,-6.5,0.7])
x_ = np.array([1,0,-0.5,-1])
y_ = np.array([0,0.5,1.5,1])

def energy(x,y,A,a,b,c,x_,y_):
    energy_ = np.zeros((x.shape))
    for i in range(len(A)):
        energy_ += A[i]*np.exp(a[i]*(x-x_[i])**2+b[i]*(x-x_[i])*(y-y_[i])+c[i]*(y-y_[i])**2)
    return energy_

energies = energy(xx,yy,A,a,b,c,x_,y_)

from scipy.integrate import simps
print(simps(simps(grad_2_zz*np.exp(-1.0*energies),points_y),points_x))

#fig, ax = plt.subplots(1,1, figsize = (7.0,2.0), dpi=600)
h = plt.contourf(xx,yy,energies,levels=[-15+i for i in range(16)])
plt.colorbar()
CS = plt.contour(points_x, points_y, zz,levels=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99], cmap='Greys')
plt.colorbar()
plt.clabel(CS, fontsize=10, inline=1)
plt.tick_params(axis='both', which='major', labelsize=9)
plt.tick_params(axis='both', which='minor', labelsize=9)
plt.savefig('committor_fem.pdf', bbox_inches='tight')
plt.close()
