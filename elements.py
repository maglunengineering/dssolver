import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm


np.set_printoptions(suppress=True)


class Problem:

    def __init__(self):
        self.nodes = list()
        self.beams = list()

        self.constrained_dofs = tuple()
        self.loads = np.array([])  # Joint loads
        self.member_loads = np.array([])  # Member loads, for distr loads and such
        self.forces = None  # Forces (at all nodes, incl removed dofs)
        self.displacements = None  # Assigned at solution ( self.solve() )
        self.solved = False

    def create_beam(self, r1, r2, E=2e5, A=1e5, I=1e5, z=None, drawnodes=3):
        """
        drawnode:
        0 - Don't draw any nodes
        1 - Draw node at r1
        2 - Draw node at r2
        3 - Draw nodes at r1 and r2
        """
        beam = Beam(r1, r2, E, A, I, z)

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

    def create_beams(self, r1, r2, E=2e5, A=1e5, I=1e5, z=None, n=4):
        nodes = np.array([ np.linspace(r1[0], r2[0], n+1),
                           np.linspace(r1[1], r2[1], n+1)])

        for ri, rj in zip(nodes.T, nodes.T[1:]):
            if np.all(ri == r1):
                self.create_beam(ri, rj, E, A, I, z, drawnodes=1)
            elif np.all(rj == r2):
                self.create_beam(ri, rj, E, A, I, z, drawnodes=2)
            else:
                self.create_beam(ri, rj, E, A, I, z, drawnodes=0)

    def create_node(self, r, draw=False):
        node = Node(r, draw=draw)
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

    def remove_node(self, r):
        self.nodes.remove(self.nodes[self.node_at(r)])

    def remove_element(self, r1, r2):
        self.beams.remove(self.beams[self.beam_at(r1, r2)])

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
            if beam == Beam(r1, r2) or beam == Beam(r2, r1):
                return beam.number
        return 'No beam'

    def fix(self, node_id):
        self.constrained_dofs += tuple(self.nodes[node_id].dofs)
        self.nodes[node_id].draw = False
        self.nodes[node_id].boundary_condition = 'fixed'

    def pin(self, node_id):
        self.constrained_dofs += tuple(self.nodes[node_id].dofs[0:2])
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'pinned'

    def roller(self, node_id):
        self.constrained_dofs += (self.nodes[node_id].dofs[1],)
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'roller'

    def lock(self, node_id):
        self.constrained_dofs += (self.nodes[node_id].dofs[2],)
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'locked'

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

    def load_member_distr(self, member_id, load):
        # Distributed load
        beam = self.beams[member_id]  # Beam object
        beam.member_loads = -beam.beta.T @ np.array([0,
                                      load * beam.length/2,
                                      load * beam.length**2 / 12,
                                      0,
                                      load*beam.length/2,
                                      -load * beam.length**2 / 12])
        beam.distributed_load = load if load != 0 else False  # For drawing purposes
        # Distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12

    def load_members_distr(self, r1, r2, load):
        # Distributed load on colinear beams from r1 to r2
        starting_node = self.nodes[self.node_at(r1)]
        ending_node = self.nodes[self.node_at(r2)]
        r1 = np.array(r1); r2 = np.array(r2)
        dir = (r2 - r1) / np.linalg.norm(r2 - r1)

        checking_node = starting_node
        while checking_node != ending_node:
            # OBS! Will loop forever if it fails
            for possible_next_node in checking_node.connected_nodes():
                dir_to_node = (possible_next_node.r - checking_node.r)/ \
                              np.linalg.norm(possible_next_node.r - checking_node.r)
                if np.isclose(dir @ dir_to_node, 1):
                    # possible_next_node is the next node
                    self.load_member_distr(self.beam_at(possible_next_node.r,
                                                        checking_node.r), load)
                    checking_node = possible_next_node

        print('Done adding loads')

    def model_size(self):
        xy = self.nodal_coordinates
        if not np.any(xy):
            return 1
        else:
            model_size = np.sqrt( (np.max(xy[:,0]) - np.min(xy[:,0]))**2 + (np.max(xy[:,1]) - np.min(xy[:,1]))**2)
            return model_size

    def K(self, reduced=False):
        dofs = 3 * len(self.nodes)  # int: Number of system dofs
        K = np.zeros((dofs,dofs))
        for beam in self.beams:
            K += beam.Ki(dofs)
        if not reduced:
            return K
        else:
            K = np.delete(K, self.constrained_dofs, axis=0)
            K = np.delete(K, self.constrained_dofs, axis=1)
            return K

    def Qf(self):  # Member force vector
        dofs = 3*len(self.nodes)
        self.member_loads = np.zeros(dofs)
        for beam in self.beams:
            self.member_loads += beam.Qfi(dofs)

    def solve(self):
        free_dofs = np.delete(np.arange(3 * len(self.nodes)), self.constrained_dofs)
        self.Qf()  # Compute system member load vector

        Kr = self.K(reduced=True)
        F = self.loads - self.member_loads
        Fr = F[free_dofs]
        print('Fr', Fr)
        print('Reduced stiffness matrix size', np.shape(Kr))
        dr = np.linalg.inv(Kr) @ Fr

        self.displacements = np.zeros(3 * len(self.nodes))
        self.displacements[free_dofs] = dr

        for node in self.nodes:
            node.displacements = self.displacements[np.array(node.dofs)]
        for beam in self.beams:
            beam.displacements = np.hstack((beam.nodes[0].displacements,
                                            beam.nodes[1].displacements))
            beam.forces = beam.k @ beam.displacements + beam.member_loads
            beam.stress = beam.cpl @ beam.beta @ beam.forces
            #beam.nodes[0].forces = beam.forces[0:3]
            #beam.nodes[1].forces = beam.forces[3:6]  # No sign convention

        print('Nodal displacements', self.displacements)
        print('Nodal loads', self.loads)
        print('Member loads', self.member_loads)

        self.forces = np.array([beam.forces for beam in self.beams])
        # forces.shape == (n, 6), n: no. of beams

        self.solved = True
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
            if not np.isclose(np.linalg.norm(node.loads[0:2]), 0):
                print('Norm', np.linalg.norm(node.loads[0:2]))
                plt.arrow(*(node.r + node.displacements[0:2]),
                          *node.loads[0:2]/np.linalg.norm(node.loads[0:2])*self.model_size()/10,
                           head_width = 20
                          )

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
        nodal_coordinates = np.array([node.r for node in self.nodes])
        return nodal_coordinates


class Node:
    def __init__(self, xy, draw=False):
        self.x, self.y = xy
        self.r = np.array(xy)

        self.beams = list()
        self.loads = np.array([0,0,0]) # self.loads (Fx, Fy, M) assigned on loading
        self.forces = np.array([0,0,0])  # self.forces (Fx, Fy, M) assigned on solution

        self.number = None  # (node id) assigned on creation
        self.dofs = None  # self.dofs (dof1, dof2, dof3) assigned on creation
        self.displacements = np.array([0,0,0])  # self.displacement (d1, d2, d3) asssigned on solution
        self.boundary_condition = None  # 'fixed', 'pinned', 'roller', 'lock'

        self.draw = draw  # Node is not drawn unless it is interesting

    def add_beam(self, beam):
        self.beams.append(beam)

    def connected_nodes(self):
        other_nodes = []
        for beam in self.beams:
            for node in beam.nodes:
                if not node == self:
                    other_nodes.append(node)
        return other_nodes

    def __str__(self):
        return '{},{}'.format(self.x, self.y)

    def __eq__(self, other):
        return np.array_equal(self.r, other.r)

class Beam:

    def __init__(self, r1, r2, E=2e5, A=1e5, I=1e5, z=None):
        self.nodes = list()

        self.E = E
        self.A = A
        self.I = I
        self.z = z if z else np.sqrt(I/A)/3

        self.r1, self.r2 = np.asarray(r1), np.asarray(r2)
        self.angle = np.arctan2( *(self.r2 - self.r1)[::-1] )
        self.length = np.sqrt( np.dot(self.r2 - self.r1, self.r2 - self.r1) )

        self.number = None  # (beam id) assigned on creation
        self.displacements = np.zeros(6)  # (1x6) assigned on solution
        self.forces = np.zeros(6)  # (1x6) assigned on solution
        self.stress = np.zeros(6)  # (1x6) assigned on solution
        # Stress given as sigma_x (top), sigma_x (bottom), tau_xy (average!)
        self.member_loads = np.zeros(6)  #
        self.distributed_load = 0
        # Distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12

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

        self.cpl = np.zeros((6,6))
        self.cpl_ = np.array([[1/self.A, 0, -self.z/self.I],
                             [1/self.A, 0, self.z/self.I],
                             [0,     1/self.A,   0]])
        self.cpl[0:3, 0:3] = self.cpl[3:6, 3:6] = self.cpl_



    def Ki(self, newdim):
        dofs = self.nodes[0].dofs + self.nodes[1].dofs
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E @ self.k @ E.T

    def Qfi(self, newdim):
        dofs = self.nodes[0].dofs + self.nodes[1].dofs
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E@self.member_loads


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
            m.create_beams((0,0),(1000,0), n=2)
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

        def lc14():
            m.create_beams((0,0),(707,-707), n=1)
            m.fix(m.node_at((0,0)))
            #m.pin(m.node_at((0,1000)))

            #m.load_node(m.node_at((500,0)), (0, -10000, 0))
            #for i in range(6):
            #    m.load_member_distr(i, 1)
            m.load_members_distr((0,0),(707,-707), 10)

        def lc15():
            m.create_beams((0,0),(1000,0), n=1)
            m.fix(m.node_at((0,0)))
            m.load_members_distr((0,0),(1000,0),10)


    lc2()

    #m.solve()
    m.plot()
    #print(m.nodes[m.node_at(n)].displacements)
    #m.plot_displaced()
    plt.show()