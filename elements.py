import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.patches as patches
import itertools as it

np.set_printoptions(suppress=True)


class Problem:

    def __init__(self):
        self.nodes = list()
        self.beams = list()

        self.constrained_dofs = tuple()
        self.loads = np.array([])
        self.displacements = None  # Assigned at solution ( self.solve() )

    def create_beam(self, r1, r2, E=2e5, A=1e5, I=1e5, drawnodes=3):
        """
        drawnode:
        0 - Don't draw any nodes
        1 - Draw node at r1
        2 - Draw node at r2
        3 - Draw nodes at r1 and r2
        """
        beam = Beam(r1, r2, E, A, I)

        if beam not in self.beams:
            for r in (r1, r2):
                node = self.create_node(r)  # Create (or identify) node
                beam.nodes.append(node)  # Add node to beam node list
                node.beams.append(beam)  # Add beam to node beam list


            self.beams.append(beam)
            beam.number = self.beams.index(beam)
            print('Beam created between', r1, r2)

            if drawnodes == 1:
                beam.nodes[0].draw = True
            elif drawnodes == 2:
                beam.nodes[1].draw = True
            elif drawnodes == 3:
                beam.nodes[0].draw = beam.nodes[1].draw = True


        else:  # Beam already exists between r1 and r2
            print('NO BEAM CREATED, already exists between', r1, r2)
            return self.beams[self.beams.index(beam)]


    def create_rod(self, r1, r2, E=2e5, A=1e5, *args):
        rod = Rod(r1, r2, E, A)

        for r in (r1, r2):
            node = self.create_node(r)  # Create (or identify) node
            rod.nodes.append(node)  # Add node to beam node list
            node.beams.append(rod)  # Add beam to node beam list

        self.beams.append(rod)
        print('Rod created between', r1, r2)

    def create_beams(self, r1, r2, E=2e5, A=1e5, I=1e5, n=10):
        nodes = np.array([ np.linspace(r1[0], r2[0], n+1),
                           np.linspace(r1[1], r2[1], n+1)])


        for ri, rj in zip(nodes.T, nodes.T[1:]):
            if np.all(ri == r1):
                self.create_beam(ri, rj, E, A, I, drawnodes=1)
            elif np.all(rj == r2):
                self.create_beam(ri, rj, E, A, I, drawnodes=2)
            else:
                self.create_beam(ri, rj, E, A, I, drawnodes=0)

    def create_node(self, r):
        node = Node(r)
        if node not in self.nodes:  # Node does not exist, i.e. this is a new node
            self.nodes.append(node)
            self.loads = np.append(self.loads, np.zeros(3))  # Append three zeroes to global load vector

            node.number = self.nodes.index(node)  # Give node a unique number/id
            node.dofs = (node.number * 3, 1 + node.number * 3, 2 + node.number * 3)
                # Assign dofs (3 pr node)

            print('Node created at {} (id {})'.format(r, node.number))
            return node
        else:  # Node already exists at r
            return self.nodes[self.nodes.index(node)] # Return the node which already exists

    def node_at(self, r):
        r = np.asarray(r)
        for node in self.nodes:
            if node == Node(r):
                node_id = node.number  # node_id: int
                return node_id
        print('No node at {}'.format(r))

    def beam_at(self, r1, r2):
        r1, r2 = np.asarray(r1), np.asarray(r2)
        for beam in self.beams:
            if beam == Beam(r1, r2):
                return beam.number
        return 'No beam'

    def fix(self, node_id):
        self.constrained_dofs += tuple(self.nodes[node_id].dofs)
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'fixed'

    def pin(self, node_id):
        self.constrained_dofs += tuple(self.nodes[node_id].dofs[0:2])
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'pinned'

    def roller(self, node_id):
        self.constrained_dofs += (self.nodes[node_id].dofs[1],)
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'roller'

    def glider(self, node_id):
        self.constrained_dofs += (self.nodes[node_id].dofs[2], self.nodes[node_id].dofs[2])
        self.nodes[node_id].draw = True
        # Broken for unknown reasons atm

    def custom(self, node_id, dofs):
        dofs = np.array(dofs)
        self.constrained_dofs += tuple(np.array(self.nodes[node_id].dofs)[dofs])
        self.nodes[node_id].draw = True

    def load_node(self, node_id, load):
        # Load : global (Nx, Ny, M)
        dofs = list(self.nodes[node_id].dofs)
        self.nodes[node_id].loads = np.array(load)
        self.loads[dofs] = load
        self.nodes[node_id].draw = True

    def model_size(self):
        xy = np.array([node.r for node in self.nodes])
        model_size = np.sqrt( (np.max(xy[:,0]) - np.min(xy[:,0]))**2 + np.max(xy[:,1]) - np.min(xy[:,1]))
        return model_size

    def K(self, reduced=False):
        dofs = 3 * len(self.nodes)
        K = np.zeros((dofs,dofs))
        for beam in self.beams:
            K += beam.Ki(dofs)
        if not reduced:
            return K
        else:
            K = np.delete(K, self.constrained_dofs, axis=0)
            K = np.delete(K, self.constrained_dofs, axis=1)
            return K

    def K_nlgeom(self, reduced=False):
        dofs = 3*len(self.nodes)
        K = np.zeros((dofs, dofs))
        for beam in self.beams:
            K += beam.Ki(dofs)
        if not reduced:
            return K
        else:
            K = np.delete(K, self.constrained_dofs, axis=0)
            K = np.delete(K, self.constrained_dofs, axis=1)
            return K

    def solve(self):
        free_dofs = np.delete(np.arange(3 * len(self.nodes)), self.constrained_dofs)

        Kr = self.K(reduced=True)
        Fr = self.loads[free_dofs]
        dr = np.linalg.inv(Kr) @ Fr

        self.displacements = np.zeros(3 * len(self.nodes))
        self.displacements[free_dofs] = dr

        for node in self.nodes:
            node.displacements = self.displacements[np.array(node.dofs)]
        print('Nodal displacements', self.displacements)
        print('Nodal loads', self.loads)
        return dr

    def plot(self):
        nodal_coordinates = np.array([0,0])

        plt.figure()
        for node in self.nodes:

            nodal_coordinates = np.vstack((nodal_coordinates, node.r))
            if node.draw:
                plt.plot(node.r[0], node.r[1], 'ko')


        for beam in self.beams:
            plt.plot(*np.array([beam.r1, beam.r2]).T, color='b')

        k = 10
        plt.xlim((np.min(nodal_coordinates[:, 0]) - self.model_size() / k,
                  np.max(nodal_coordinates[:, 0]) + self.model_size() / k))
        plt.ylim((np.min(nodal_coordinates[:, 1]) - self.model_size() / k,
                  np.max(nodal_coordinates[:, 1]) + self.model_size() / k))

        try:
            plt.arrow(*node.r,
                      *node.loads[0:2]/np.linalg.norm(node.loads[0:2])*self.model_size()/10,
                      head_width=20 )
        except:
            pass

    def plot_displaced(self, scale=1):
        nodal_coordinates = np.array([0, 0])

        plt.figure()
        for node in self.nodes:
            nodal_coordinates = np.vstack((nodal_coordinates, node.r))

            if node.draw:
                plt.plot(node.r[0] + node.displacements[0]*scale,
                         node.r[1] + node.displacements[1]*scale, 'ko')
            try:
                plt.arrow(*(node.r + node.displacements[0:2]),
                          *node.loads[0:2]/np.linalg.norm(node.loads[0:2])*self.model_size()/10,
                           head_width = 20
                          )
            except:
                pass
        for beam in self.beams:
            pos = np.array([beam.r1, beam.r2])
            disp = np.array([beam.nodes[0].displacements[0:2]*scale,
                             beam.nodes[1].displacements[0:2]*scale])

            plt.plot(*(pos+disp).T, color='b')

        k = 10
        plt.xlim((np.min(nodal_coordinates[:, 0]) - self.model_size() / k,
                  np.max(nodal_coordinates[:, 0]) + self.model_size() / k))
        plt.ylim((np.min(nodal_coordinates[:, 1]) - self.model_size() / k,
                  np.max(nodal_coordinates[:, 1]) + self.model_size() / k))

    def animate(self, n=10):
        fig = plt.figure()
        ax = fig.gca()

        def animate_(k):
            scale = k/n
            for node in self.nodes:
                nodal_coordinates = np.vstack((nodal_coordinates, node.r))
                ax.plot(node.r[0] + node.displacements[0]*scale,
                         node.r[1] + node.displacements[1]*scale, 'ko')
            for beam in self.beams:
                pos = np.array([beam.r1, beam.r2])
                disp = np.array([beam.nodes[0].displacements[0:2]*scale,
                                 beam.nodes[1].displacements[0:2]*scale])

                ax.plot(*(pos + disp).T, color='b')


        anim = anm.FuncAnimation(fig, animate_, frames=n, interval=100, blit=False)

    @property
    def nodal_coordinates(self):
        nodal_coordinates = np.array([0,0])
        for node in self.nodes:
            nodal_coordinates = np.vstack((nodal_coordinates, node.r))
        return nodal_coordinates



class Node:
    def __init__(self, xy):
        self.x, self.y = xy
        self.r = xy

        self.beams = list()
        self.loads = np.array([0,0,0]) # self.loads (Fx, Fy, M) assigned on loading

        self.number = None  # (node id) assigned on creation
        self.dofs = None  # self.dofs (dof1, dof2, dof3) assigned on creation
        self.displacements = None  # self.displacement (d1, d2, d3) asssigned on solution
        self.boundary_condition = None  # 'fixed', 'pinned', 'roller'

        self.draw = False  # Node is not drawn unless it is interesting

    def add_beam(self, beam):
        self.beams.append(beam)

    def __str__(self):
        return '{},{}'.format(self.x, self.y)

    def __eq__(self, other):
        return np.array_equal(self.r, other.r)

class Beam:

    def __init__(self, r1, r2, E=2e5, A=1e5, I=1e5):
        self.nodes = list()

        self.r1, self.r2 = np.asarray(r1), np.asarray(r2)
        self.angle = np.arctan2( *(self.r2 - self.r1)[::-1] )
        self.length = np.sqrt( np.dot(self.r2 - self.r1, self.r2 - self.r1) )

        self.number = None  # (beam id) assigned on creation

        self.s, self.c = np.sin(self.angle), np.cos(self.angle)
        self.beta = np.array([[self.c, self.s, 0, 0, 0, 0],
                              [-self.s, self.c, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, self.c, self.s, 0],
                              [0, 0, 0, -self.s, self.c, 0],
                              [0, 0, 0, 0, 0, 1]])

        
        self.kn = A*E/self.length * (E*I/self.length**3)**(-1)
        self.ke = E*I/self.length**3 * np.array(
             [[self.kn, 0, 0, -self.kn, 0, 0],
              [0, 12, 6*self.length, 0, -12, 6*self.length],
              [0, 6*self.length, 4*self.length**2, 0, -6*self.length, 2*self.length**2],
              [-self.kn, 0, 0, self.kn, 0, 0],
              [0, -12, -6*self.length, 0, 12, -6*self.length],
              [0, 6*self.length, 2*self.length**2, 0, -6*self.length, 4*self.length**2]])

        self.k = self.beta.T @ self.ke @ self.beta

    def Ki(self, newdim):
        dofs = self.nodes[0].dofs + self.nodes[1].dofs
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E @ self.k @ E.T


    def __eq__(self, other):
        return np.array_equal(np.array([self.r1, self.r2]), np.array([other.r1, other.r2]))

class Rod(Beam):
    """

    A beam element with no bending or shear stiffness.
    However, the endnodes of this element must have stiffness against bending and shear, or the stiffness
     matrix will be singular.
    This means both endnodes must either be connected to a beam element or be restrained (e.g. fixed).
    """


    def __init__(self, r1, r2, E=2e5, A=1e5, **kwargs):
        super().__init__(r1=r1, r2=r2, E=E, A=A)

        self.ke = np.array(      [[self.kn, 0, 0, -self.kn, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [-self.kn, 0, 0, self.kn, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]])

        self.k = self.beta.T @ self.ke @ self.beta

if __name__ == '__main__':
    m = Problem()

    for _ in range(1):
        def lc1():
            m.create_beams((0,0),(0,1000), n=10)
            m.create_beams((0,1000), (1000,1000), n=10)
            m.create_beams((0,1000), (-500,1500), n=5, I = 2e3)

            m.fix(0)
            m.pin(m.node_at((1000,1000)))

            #m.load_node(m.node_at((-500,1500)), (200, 200, 0))
            #m.load_node(m.node_at((0, 1000)), (0, 0, 500000))
            m.load_node(m.node_at((1000,1000)), (0, 0, 50000000))

        def lc2():
            m.create_beams((0,0),(1000,0), n=10)
            m.fix(m.node_at((0,0)))
            #m.pin(m.node_at((1000,0)))

            #m.load_node(m.node_at((0,0)), (0, 0, 100000))
            m.load_node(m.node_at((1000,0)), (0, -1000, 0))
            n = (1000, 0)
            return n

        def lc3():
            m.create_beams((0,0), (0,1000), n=10)
            m.create_beams((0,1000), (1000,1000), n=10)

            m.pin(m.node_at((0,0)))
            m.pin(m.node_at((1000,1000)))

            m.load_node(m.node_at((500,1000)), (0, -100000, 0))
            m.load_node(m.node_at((0, 500)), (-100000, 0, 0))

        def lc4():
            m.create_beams((0,0), (1000,0), n=10)
            m.create_beams((1000,0), (1000,1000), n=10, I=10)
            m.create_rod((1000,1000), (2000,1000), A=1)
            #m.create_rod((1500,1000), (2000,1000), A=1)

            m.pin(m.node_at((0,0)))
            m.roller(m.node_at((1000,0)))
            m.fix(m.node_at((2000,1000)))

            m.load_node(m.node_at((500,0)), (0, -150000, 0))

        def lc5():
            m.create_beams((0,0),(1000,0), n=20)
            m.create_rod((500,-1000), (500,0))

            m.pin(m.node_at((0,0)))
            m.pin(m.node_at((1000,0)))
            m.fix(m.node_at((500,-1000)))

            m.load_node(m.node_at((300,0)), (0, -1000000, 0))
            m.load_node(m.node_at((700,0)), (0, -1000000, 0))

        def lc6():
            m.create_beams((0,0), (500,0))
            m.create_beams((500,0), (500, 1000), n=40)
            m.create_beams((500,1000), (0, 1000), n=40)
            m.create_beams((0, 1000), (0,0))

            m.load_node(m.node_at((0,500)), (100000,0,0))
            m.load_node(m.node_at((500,500)), (-100000,0,0))

            m.pin(m.node_at((0,0)))
            m.pin(m.node_at((500,0)))
            m.custom(m.node_at((0,700)), (0,2))

            return (0,0)

        def lc7():
            m.create_beams((0,0),(1000,0), n=5)
            m.create_beams((1000,0), (1000,-1000), n=5)

            m.fix(m.node_at((0,0)))

            m.load_node(m.node_at((1000,-1000)), (-1000, 0, 0))
            n = (1000,-1000)
            return n

        def lc8():
            m.create_beams((0,0),(1000,0), n=10)

            m.fix(m.node_at((0,0)))
            m.pin(m.node_at((1000,0)))

            m.load_node(m.node_at((1000,0)), (0,0,1e7))

        def lc9():
            m.create_beams((0,0),(1000,0), n=20)
            m.fix(m.node_at((0,0)))
            m.fix(m.node_at((1000,0)))

            m.load_node(m.node_at((500,0)), (0, -100000,0))

        def lc10():
            m.create_beams((0,0),(1000,0), n=5)
            m.create_beams((0,-150),(700,-150), n=5)
            m.create_beams((700,-150), (1000,0), n=5)
            m.create_beams((1000,0),(2000,0), n=5)
            m.fix(m.node_at((0,0)))
            m.fix(m.node_at((0,-150)))

            m.load_node(m.node_at((2000,0)), (0,-10000,0))

        def lc11():
            m.create_beams((0,0),(1000,0), n=20)
            m.fix(m.node_at((0,0)))
            m.pin(m.node_at((1000,0)))
            m.load_node(m.node_at((1000,0)), (0, 0, -1e7))

        def lc12():
            m.create_beams((0,0),(500,0), n=5)
            m.create_rod((500,0), (1000,0))
            m.fix(m.node_at((0,0)))
            m.glider(m.node_at((1000,0)))
            m.roller(m.node_at((1000,0)))
            m.load_node(m.node_at((1000,0)), (1e6, 0, 0))

        def lc13():
            m.create_beams((20,120), (200,120))
            m.create_beams((200,120), (200,240))
            m.fix(m.node_at((20,120)))
            m.load_node(m.node_at((200,240)), (100000,0,0))


    lc6()

    m.solve()
    #m.plot()
    #print(m.nodes[m.node_at(n)].displacements)
    m.plot_displaced()
    plt.show()