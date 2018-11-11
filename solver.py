import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import time 

np.set_printoptions(suppress=True)


class Problem:

    def __init__(self):
        self.nodes = list()
        self.beams = list()

        self.constrained_dofs = np.array([])
        self.loads = np.array([])  # Joint loads
        self.member_loads = np.array([])  # Member loads, for distr loads and such
        self.forces = None  # Forces (at all nodes, incl removed dofs)
        self.displacements = None  # Assigned at solution ( self.solve() )
        self.solved = False

        # Nonlinear analysis variables
        self.displ_history = None
        self.load_history = None
        self.A_history = None

        self.max_increments = 200
        self.fwde_increments = 100
        self.max_iterations = 25
        self.arclength = 50
        self.totalarclength = 10000
        self.res_norm_criterion = 0.1
        self.stiffness_upd_interval = 2

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

    def create_rod(self, r1, r2, E=2e5, A=1e5, *args, **kwargs):
        rod = Rod(r1, r2, E, A)

        for r in (r1, r2):
            node = self.create_node(r)  # Create (or identify) node
            rod.nodes.append(node)  # Add node to beam node list
            node.beams.append(rod)  # Add beam to node beam list

        self.beams.append(rod)
        rod.number = self.beams.index(rod)
        print('Rod created between', r1, r2)

    def create_beams(self, r1, r2, E=2e5, A=1e5, I=1e5, z=None, n=4):
        nodes = np.array([np.linspace(r1[0], r2[0], n + 1),
                          np.linspace(r1[1], r2[1], n + 1)])

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
            return self.nodes[self.nodes.index(node)]  # Return the node which already exists

    def remove_orphan_nodes(self):
        for node in self.nodes.copy():
            if len(node.beams) == 0:
                self.nodes.remove(node)
        self.reassign_dofs()

    def remove_element(self, r1, r2):
        try:
            element = self.beams[self.beam_at(r1, r2)]
            for node in element.nodes:
                node.beams.remove(element)
            self.beams.remove(self.beams[self.beam_at(r1, r2)])
            print('Removed element at', r1, r2)
            self.remove_orphan_nodes()
        except:
            print('Failed to remove; No element at', r1, r2)

    def node_at(self, r):
        r = np.asarray(r)
        for node in self.nodes:
            if node == Node(r):
                node_id = node.number  # node_id: int
                return node_id
        print('No node at {}'.format(r))

    def nodeobj_at(self, r):
        r = np.asarray(r)
        for node in self.nodes:
            if node == Node(r):
                return node
        print('No node at {}'.format(r))

    def free_dofs(self):
        return np.delete(np.arange(3 * len(self.nodes)), self.constrained_dofs)

    def beam_at(self, r1, r2):
        r1, r2 = np.asarray(r1), np.asarray(r2)
        for beam in self.beams:
            if beam == Beam(r1, r2) or beam == Beam(r2, r1):
                return self.beams.index(beam)

        print('No beam at', r1, r2)

    def reassign_dofs(self):
        self.loads = np.array([])
        for i, node in enumerate(self.nodes):
            node.number = i
            node.dofs = (3 * i, 3 * i + 1, 3 * i + 2)
            self.loads = np.append(self.loads, node.loads)

    def reform_geometry(self):
        for element in self.beams:
            new_r1 = element.r1 + element.displacements[0:2]
            new_r2 = element.r2 + element.displacements[3:5]

            deformed_angle = np.arccos((np.dot(new_r2 - new_r1, np.array([1, 0])) /
                                        np.linalg.norm(new_r2 - new_r1)))
            element.k = beta(deformed_angle).T @ element.ke @ beta(deformed_angle)
            # print('Deformation angle', deformed_angle)

    def upd_elements(self):
        for node in self.nodes:
            node.displacements = self.displacements[np.array(node.dofs)]
        for beam in self.beams:
            beam.displacements = u = self.displacements[beam.dofs]
            beam.forces = beam.k @ u + beam.member_loads
            beam.stress = beam.cpl @ beam.beta(beam.angle) @ beam.forces

            ## --------------- CR formulation
            r1d = beam.r1 + u[0:2]
            r2d = beam.r2 + u[3:5]
            deformed_length = beam.deformed_length()
            un = deformed_length - beam.length

            tan_e = beam.tangent
            tan_ed = (r2d - r1d) / deformed_length
            tan_1 = R(u[2]) @ tan_e
            tan_2 = R(u[5]) @ tan_e

            theta1, theta2 = np.arcsin(np.array([tan_ed[0]*tan_1[1] - tan_ed[1]*tan_1[0],
                                                 tan_ed[0]*tan_2[1] - tan_ed[1]*tan_2[0]]))

            beam.displacements_local = np.array([-un / 2, 0, theta1, un / 2, 0, theta2])
            beam.forces_local = beam.ke @ beam.displacements_local

    def boundary_condition(self, r, bctype):
        """
        :param r: Location of node
        :param bctype: 'fixed', 'pinned', 'roller', 'lock'
        """
        r = np.asarray(r)
        if bctype == 'fixed':
            self.fix(self.node_at(r))
        elif bctype == 'pinned':
            self.pin(self.node_at(r))
        elif bctype == 'roller':
            self.roller(self.node_at(r))
        elif bctype == 'lock':
            self.lock(self.node_at(r))
        elif bctype == 'glider':
            self.glider(self.node_at(r))

    def fix(self, node_id):
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'fixed'

    def pin(self, node_id):
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'pinned'

    def roller(self, node_id):
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'roller'

    def lock(self, node_id):
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'locked'

    def glider(self, node_id):
        self.nodes[node_id].draw = True
        self.nodes[node_id].boundary_condition = 'glider'

    def custom(self, node_id, dofs):
        dofs = np.array(dofs)
        self.constrained_dofs += tuple(np.array(self.nodes[node_id].dofs)[dofs])
        self.nodes[node_id].draw = True

    def auto_rotation_lock(self):
        """
        Rotation locks all nodes where only Rod elements meet. Useful for truss analysis.
        """
        for node in self.nodes:
            if np.all([type(element) == Rod for element in node.beams]) \
                    and node.boundary_condition is None:
                self.lock(node.number)

    def remove_dofs(self):  # Interpret boundary conditions
        self.constrained_dofs = []
        for node in self.nodes:
            if node.boundary_condition == 'fixed':
                self.constrained_dofs.extend(node.dofs)
            elif node.boundary_condition == 'pinned':
                self.constrained_dofs.extend(node.dofs[0:2])
            elif node.boundary_condition == 'roller':
                self.constrained_dofs.append(node.dofs[1])
            elif node.boundary_condition == 'locked':
                self.constrained_dofs.append(node.dofs[2])
            elif node.boundary_condition == 'glider':
                self.constrained_dofs.extend((node.dofs[0], node.dofs[2]))

    def load_node(self, node_id, load):
        # Load : global (Nx, Ny, M)
        dofs = list(self.nodes[node_id].dofs)
        self.nodes[node_id].loads = np.array(load)
        self.loads[dofs] = load
        self.nodes[node_id].draw = True

    def load_member_distr(self, member_id, load):
        # Distributed load
        beam = self.beams[member_id]  # Beam object
        beam.member_loads = -beam.beta(beam.angle).T @ np.array([0,
                                                                 load * beam.length / 2,
                                                                 load * beam.length ** 2 / 12,
                                                                 0,
                                                                 load * beam.length / 2,
                                                                 -load * beam.length ** 2 / 12])
        beam.distributed_load = load if load != 0 else False  # For drawing purposes
        # Distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12

    def load_members_distr(self, r1, r2, load):
        # Distributed load on colinear beams from r1 to r2
        starting_node = self.nodes[self.node_at(r1)]
        ending_node = self.nodes[self.node_at(r2)]
        r1 = np.array(r1);
        r2 = np.array(r2)
        dir = (r2 - r1) / np.linalg.norm(r2 - r1)

        checking_node = starting_node
        while checking_node != ending_node:
            # OBS! Will loop forever if it fails
            for possible_next_node in checking_node.connected_nodes():
                dir_to_node = (possible_next_node.r - checking_node.r) / \
                              np.linalg.norm(possible_next_node.r - checking_node.r)
                if np.isclose(dir @ dir_to_node, 1):
                    # possible_next_node is in fact the next node
                    self.load_member_distr(self.beam_at(possible_next_node.r,
                                                        checking_node.r), load)
                    checking_node = possible_next_node

        print('Done adding loads')

    def model_size(self):
        xy = self.nodal_coordinates
        if not np.any(xy):
            return 1
        else:
            model_size = np.sqrt(
                (np.max(xy[:, 0]) - np.min(xy[:, 0])) ** 2 + (np.max(xy[:, 1]) - np.min(xy[:, 1])) ** 2)
            return model_size

    def K(self, reduced=False):
        self.remove_dofs()
        no_of_dofs = 3 * len(self.nodes)  # int: Number of system dofs
        K = sum(element.Ki(no_of_dofs)
                + element.Ex(no_of_dofs) @ element.kG() @ element.Ex(no_of_dofs).T
                for element in self.beams)
        if not reduced:
            return K
        else:
            K = np.delete(K, self.constrained_dofs, axis=0)
            K = np.delete(K, self.constrained_dofs, axis=1)
            return K

    def Qf(self):  # Member force vector for distr loads
        dofs = 3 * len(self.nodes)
        self.member_loads = np.zeros(dofs)
        for beam in self.beams:
            self.member_loads += beam.Qfi(dofs)

    def solve(self):
        self.reassign_dofs()
        self.remove_dofs()
        free_dofs = np.delete(np.arange(3 * len(self.nodes)), self.constrained_dofs)
        self.Qf()  # Compute system load vector

        Kr = self.K(reduced=True)
        F = self.loads - self.member_loads
        Fr = F[free_dofs]
        print('Fr', Fr)
        print('Reduced stiffness matrix size', np.shape(Kr))
        dr = np.linalg.solve(Kr, Fr)

        self.displacements = np.zeros(3 * len(self.nodes))
        self.displacements[free_dofs] = dr
        # self.ext_forces = self.K() @ self.displacements

        self.upd_elements()

        print('Nodal displacements', self.displacements)
        print('Nodal loads', self.loads)
        print('Member loads', self.member_loads)

        self.forces = np.array([beam.forces for beam in self.beams])
        # forces.shape == (n, 6), n: no. of beams

        self.solved = True
        return dr

    def solve_newton(self):
        self.reassign_dofs()
        self.remove_dofs()
        A = 0
        dA = 0.01
        i = 0 # Counter

        q = self.loads
        self.displ_history = np.zeros((int(1/dA), 3*len(self.nodes)))
        self.load_history = np.zeros((int(1/dA), 3*len(self.nodes)))
        self.displacements = np.zeros(3 * len(self.nodes))
        self.A_history = []

        free_dofs = self.free_dofs()

        while A < 1:
            A += dA
            i += 1

            K = self.K(reduced=True)
            ddisp = np.linalg.solve(K, A*q[free_dofs] - self.internal_forces())
            self.displacements[free_dofs] = self.displacements[free_dofs] + ddisp

            self.upd_elements()
            res = 2 * self.res_norm_criterion

            k = 0
            while np.linalg.norm(res) > self.res_norm_criterion and k < self.max_iterations:
                res = A*q[free_dofs] - self.internal_forces()
                print(i, k, np.linalg.norm(res))
                ddisp_ = np.linalg.solve(self.K(reduced=True), res)
                self.displacements[free_dofs] = self.displacements[free_dofs] + ddisp_
                self.upd_elements()
                k += 1

            if k >= self.max_iterations:
                print("Failed to converge solution at increment {} after {} iterations".format(
                    i, self.max_iterations))
                break

            if i >= self.displ_history.shape[0]:
                self.displ_history = np.vstack((self.displ_history, np.zeros((1, 3 * len(self.nodes)))))
                self.load_history = np.vstack((self.load_history, np.zeros((1, 3 * len(self.nodes)))))
            self.displ_history[i] = self.displacements
            self.load_history[i] = (q * A)
            self.A_history.append(A)

    def solve_arclength(self, autoarclength=True):
        self.reassign_dofs()
        self.remove_dofs()
        A = 0.0
        i = 0
        j = 0
        arclength = self.arclength
        total_arclength = self.totalarclength
        imax = self.max_increments
        Amax = np.linalg.norm(self.loads)

        y = 1
        q = y * self.loads / np.linalg.norm(self.loads)
        self.displ_history = np.zeros((100, 3*len(self.nodes)))
        self.load_history = np.zeros((100, 3*len(self.nodes)))
        self.displacements = np.zeros(3 * len(self.nodes))
        self.A_history = []

        free_dofs = self.free_dofs()
        start_time = time.time()

        while np.abs(A) < Amax and i < imax:
            i += 1
            j += 1

            wq0 = np.linalg.solve(self.K(reduced=True), q[free_dofs])
            f = np.sqrt(1 + wq0@wq0)
            sign = np.sign(wq0@v0 if i > 1 else 1)
            dA = arclength / f * sign
            v0 = dA * wq0

            A += dA
            self.displacements[free_dofs]  = self.displacements[free_dofs] + v0
            self.upd_elements()


            # Corrector
            k = 0
            res = self.internal_forces() - q[free_dofs]*A

            while np.linalg.norm(res) > self.res_norm_criterion and k < self.max_iterations:
                if k % self.stiffness_upd_interval == 0:
                    K = self.K(reduced=True)
                wq = np.linalg.solve(K, q[free_dofs])
                wr = np.linalg.solve(K, -res)
                dA_ = - wq@wr / (1 + wq@wq)
                A += dA_
                #self.A_history.append(A)
                self.displacements[free_dofs] = self.displacements[free_dofs] + wr + dA_ * wq
                self.upd_elements()
                res = self.internal_forces() - q[free_dofs] * A
                k += 1
                j += 1


            print('A = {}, k = {}, i = {}/{}'
                  .format(A, k, i, imax))
            if k >= 25:
                print('Failed to converge at step {}, reverting and reducing arc length'.format(i))
                self.displacements = self.displ_history[i-1][self.free_dofs()]
                self.upd_elements()
                A = self.A_history.pop(-1)
                i -= 1
                arclength = arclength * 0.8

            if i >= self.displ_history.shape[0]:
                self.displ_history = np.vstack((self.displ_history, np.zeros((20, 3 * len(self.nodes)))))
                self.load_history = np.vstack((self.load_history, np.zeros((20, 3 * len(self.nodes)))))
            self.displ_history[i] = self.displacements
            self.load_history[i] = (q * A)
            self.A_history.append(A)

        end_time = time.time()
        time_spent = end_time - start_time
        print('Finished', i, 'increments in', time_spent, 'seconds')

    def internal_forces(self, reduced=True):
        f_int = np.zeros(3 * len(self.nodes))
        for element in self.beams:
            f_element = element.Ex(3 * len(self.nodes)) @ element.beta().T @ element.forces_local
            f_int = f_int + f_element
        if reduced:
            return f_int[self.free_dofs()]
        else:
            return f_int

    def plot(self):
        nodal_coordinates = np.array([0, 0])

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
                      *node.loads[0:2] / np.linalg.norm(node.loads[0:2]) * self.model_size() / 10,
                      head_width=20)
        except:
            pass

    def plot_displaced(self, scale=1):
        nodal_coordinates = np.array([0, 0])

        plt.figure()
        for node in self.nodes:
            nodal_coordinates = np.vstack((nodal_coordinates, node.r))

            if node.draw:
                plt.plot(node.r[0] + node.displacements[0] * scale,
                         node.r[1] + node.displacements[1] * scale, 'ko')
            if not np.isclose(np.linalg.norm(node.loads[0:2]), 0):
                print('Norm', np.linalg.norm(node.loads[0:2]))
                plt.arrow(*(node.r + node.displacements[0:2]),
                          *node.loads[0:2] / np.linalg.norm(node.loads[0:2]) * self.model_size() / 10,
                          head_width=20
                          )

        for beam in self.beams:
            pos = np.array([beam.r1, beam.r2])
            disp = np.array([beam.nodes[0].displacements[0:2] * scale,
                             beam.nodes[1].displacements[0:2] * scale])

            plt.plot(*(pos + disp).T, color='b')

        k = 10
        plt.xlim((np.min(nodal_coordinates[:, 0]) - self.model_size() / k,
                  np.max(nodal_coordinates[:, 0]) + self.model_size() / k))
        plt.ylim((np.min(nodal_coordinates[:, 1]) - self.model_size() / k,
                  np.max(nodal_coordinates[:, 1]) + self.model_size() / k))

    def animate(self, save=False, dofs=None):
        """
        :param dofs: 2-tuple (loaddof, displacementdof) global non-reduced dofs
            """
        plt.close('all')
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

        # Force-displacement
        if dofs is None:
            dofs = 2 * tuple(np.where(self.loads == np.max(np.abs(self.loads)))[0],)
        loads = self.A_history
        displ = self.displ_history[:,dofs[1]][~np.all( self.displ_history==0, axis=1)] * \
                (np.sign(self.displ_history[5,dofs[1]]))

        ax1.plot(displ, loads)
        ax1.set_title('Load - displacement')
        ax2.set_title('Animation')
        ax1.set_xlabel('Displacement' if dofs is None else 'Displacement at DOF {}'.format(dofs[1]))
        ax1.set_ylabel('Load factor')

        # Animation

        id = self.displ_history
        id = id.reshape((-1, len(self.nodes), 3))[:, :, 0:2][~np.all( id==0, axis=1)]
        no_of_frames = id.shape[0]
        interval = 20*len(loads)/no_of_frames
        #il = self.load_history.reshape((-1, len(self.nodes), 3))

        ax2.set_xlim(-1000, 1000)
        ax2.set_ylim(-600, 1400)

        line, = ax2.plot([], [])
        marker1, = ax1.plot([], [], '.', color='red', markerSize=10)
        marker2, = ax2.plot([], [], '.', color='blue', markerSize=7)
        arrow, = ax2.plot(0, 0, color='black', marker='v')
        line.set_data(self.nodal_coordinates.T)
        txt = ax2.text(-900,200, 'Frame {}')
        txt.set_text('Frame {}')

        def animate_(k):
            xb, yb = (self.nodal_coordinates[int(np.floor(dofs[1]/3))] + id[k, int(np.floor(dofs[1]/3))])
            line.set_data(self.nodal_coordinates.T + id[k].T)
            marker2.set_data(xb, yb)
            marker1.set_data(displ[k], loads[k])
            marker_dir = 'v' if loads[k] > 0 else '^'
            arrow.set_data([xb, xb], [yb, yb - 250*loads[k]/np.max(np.abs(loads))])
            arrow.set_markevery([1])
            arrow.set_marker(marker_dir)


            txt.set_text('Frame {}/{}'.format(k,no_of_frames))

        self.anim = anm.FuncAnimation(fig, animate_,
                                 frames=id.shape[0], interval=interval, blit=False)
        if save:
            Writer = anm.writers['ffmpeg']
            writer = Writer(fps=8, metadata=dict(artist='maglun engineering DSSolver'),
                            bitrate=1800)
            self.anim.save('dssolver.mp4')
        plt.show()

    def open_problem(self):
        pass

    def save_problem(self):
        pass

    @property
    def nodal_coordinates(self):
        nodal_coordinates = np.array([node.r for node in self.nodes])
        return nodal_coordinates

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

class Node:
    def __init__(self, xy, draw=False):
        self.x, self.y = xy
        self.r = np.array(xy)

        self.beams = list()
        self.loads = np.array([0, 0, 0])  # self.loads (global Fx, Fy, M) assigned on loading
        # self.forces = np.array([0,0,0])  # self.forces (Fx, Fy, M) calculated by self.abs_force

        self.number = None  # (node id) assigned on creation
        self.dofs = None  # self.dofs (dof1, dof2, dof3) assigned on creation
        self.displacements = np.array([0, 0, 0])  # self.displacement (d1, d2, d3) asssigned on solution
        self.boundary_condition = None  # 'fixed', 'pinned', 'roller', 'locked', 'glider'

        self.draw = draw  # Node is not drawn unless it is interesting

    def add_beam(self, beam):
        self.beams.append(beam)

    @property
    def abs_forces(self):
        self.forces = np.array([0, 0, 0])
        for element in self.beams:
            if np.array_equal(element.r1, self.r):
                self.forces = self.forces + np.abs(element.forces[0:3]) / 2
            elif np.array_equal(element.r2, self.r):
                self.forces = self.forces + np.abs(element.forces[3:6]) / 2
        return self.forces + np.abs(self.loads / 2)

    def connected_nodes(self):
        other_nodes = []
        for beam in self.beams:
            for node in beam.nodes:
                if not node == self:
                    other_nodes.append(node)
        return other_nodes

    def translate(self, dx, dy):
        self.x, self.y = self.x + dx, self.y + dy
        self.r = np.array([self.x, self.y])

    def __str__(self):
        return '{},{}'.format(self.x, self.y)

    def __eq__(self, other):
        return np.allclose(self.r, other.r)

class Beam:

    def __init__(self, r1, r2, E=2e5, A=1e5, I=1e5, z=None):
        self.nodes = list()

        self.E = E
        self.A = A
        self.I = I
        self.z = z if z else np.sqrt(I / A) / 3

        self.r1, self.r2 = np.asarray(r1), np.asarray(r2)
        self.angle = np.arctan2(*(self.r2 - self.r1)[::-1])
        self.length = np.sqrt(np.dot(self.r2 - self.r1, self.r2 - self.r1))
        self.tangent = (self.r2 - self.r1) / self.length

        self.number = None  # (beam id) assigned on creation
        self.displacements = np.zeros(6)  # global csys (1x6) assigned on solution
        self.displacements_local = np.zeros(6) 
        self.forces = np.zeros(6)  # global csys (1x6) assigned on solution
        self.forces_local = np.zeros(6) 
        self.stress = np.zeros(6)  # local csys(1x6) assigned on solution
        # Stress given as sigma_x (top), sigma_x (bottom), tau_xy (average!)
        self.member_loads = np.zeros(6)  # local csys distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12
        self.distributed_load = 0
        # Distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12

        self.kn = A * E / self.length * (E * I / self.length ** 3) ** (-1)
        self.ke = E * I / self.length ** 3 * np.array(
            [[self.kn, 0, 0, -self.kn, 0, 0],
             [0, 12, 6 * self.length, 0, -12, 6 * self.length],
             [0, 6 * self.length, 4 * self.length ** 2, 0, -6 * self.length, 2 * self.length ** 2],
             [-self.kn, 0, 0, self.kn, 0, 0],
             [0, -12, -6 * self.length, 0, 12, -6 * self.length],
             [0, 6 * self.length, 2 * self.length ** 2, 0, -6 * self.length, 4 * self.length ** 2]])

        self.k = self.beta(self.angle).T @ self.ke @ self.beta(self.angle)

        self.cpl = np.zeros((6, 6))
        self.cpl_ = np.array([[1 / self.A, 0, -self.z / self.I],
                              [1 / self.A, 0, self.z / self.I],
                              [0, 1 / self.A, 0]])
        self.cpl[0:3, 0:3] = self.cpl[3:6, 3:6] = self.cpl_

    def beta(self, angle=0):
        e1 = ((self.r2 + self.displacements[3:5]) -
              (self.r1 + self.displacements[0:2])) / self.deformed_length()
        e2 = np.array([-e1[1], e1[0]])
        return np.array([[*e1, 0, *np.zeros(3)],
                      [*e2, 0, *np.zeros(3)],
                      [0, 0, 1, *np.zeros(3)],
                      [*np.zeros(3), *e1, 0],
                      [*np.zeros(3), *e2, 0],
                      [*np.zeros(3), 0, 0, 1]])

    def Ki(self, newdim):
        dofs = self.nodes[0].dofs + self.nodes[1].dofs
        E = self.Ex(newdim)
        T = self.beta()
        return E @ T.T @ self.ke @ T @ E.T

    def kG(self, symmetric=False):
        fx1,fy1,m1,fx2,fy2,m2 = self.forces_local
        force_permuted = np.array([-fy1, fx1, 0, -fy2, fx2, 0])
        Ld = self.deformed_length()
        G = np.array([0, -1/Ld, 0, 0, 1/Ld, 0])
        T = self.beta()
        if symmetric:
            kg = np.outer(force_permuted, G)
            return 1/2 * (kg + kg.T)
            return 1/2 * T.T @ (kg + kg.T) @ T
        else:
            return np.outer(force_permuted, G)
            return T.T @ np.outer(force_permuted, G) @ T

    def Qfi(self, newdim):
        dofs = self.nodes[0].dofs + self.nodes[1].dofs
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E @ self.member_loads

    def Ex(self, newdim):
        dofs = self.nodes[0].dofs + self.nodes[1].dofs
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E

    def deformed_length(self):
        return np.linalg.norm((self.r2 + self.displacements[3:5] -
                        self.r1 - self.displacements[0:2]))

    def translate(self, translation, loaded=True):
        # translation: global vector [u1,v1,theta1, u2,v2,theta2]
        self.nodes[0].translate(translation[0], translation[1])
        self.nodes[1].translate(translation[3], translation[4])
        if loaded:
            self.forces = self.forces + self.k @ translation
            self.displacements = self.displacements + translation
        tr_local = beta(self.angle).T @ translation
        angle_change = np.arcsin((tr_local[4] - tr_local[1]) / self.length)
        self.k = beta(angle_change).T @ self.k @ beta(angle_change)

    def strain_energy(self, delta_displacements=np.zeros(6)):
        """
        :param delta_displacements: Additional displacements for strain energy calculations.
        The derivative of strain_energy wrt. displacement 2 is thus (strain_energy([0,0,dd,0,0,0])-strain_energy)/dd
        """
        N1, V1, M1, N2, V2, M2 = self.ke @ beta(self.angle) @ (self.displacements + delta_displacements)
        L = self.length
        return 1 / (2 * self.E * self.I) * (M1 ** 2 * L - M1 * V1 * L ** 2 + V1 ** 2 * L ** 3 / 3) + \
               (N2 ** 2 * L) / (2 * self.E * self.A)

    def strain_energy_gradient(self, dui=1e-10):
        p = np.zeros(6)
        for k in range(6):
            delta_disp = np.zeros(6)
            delta_disp[k] = dui
            p[k] = (self.strain_energy(delta_disp) - self.strain_energy()) / dui
        return p

    @property
    def dofs(self):
        return np.array(self.nodes[0].dofs + self.nodes[1].dofs)

    def __eq__(self, other):
        return np.allclose(np.array([self.r1, self.r2]), np.array([other.r1, other.r2]))

class Rod(Beam):
    """
    A beam element with no bending or shear stiffness.
    However, the endnodes of this element must have stiffness against bending and shear, or the stiffness
     matrix will be singular.
    This means both endnodes must either be connected to a beam element or be restrained (e.g. fixed).
    """

    def __init__(self, r1, r2, E=2e5, A=1e5, *args, **kwargs):
        super().__init__(r1=r1, r2=r2, E=E, A=A)

        self.kn = A * E / self.length
        self.ke = np.array([[self.kn, 0, 0, -self.kn, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [-self.kn, 0, 0, self.kn, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]])

        self.k = self.beta(self.angle).T @ self.ke @ self.beta(self.angle)

def beta(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, s, 0, 0, 0, 0],
                     [-s, c, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, c, s, 0],
                     [0, 0, 0, -s, c, 0],
                     [0, 0, 0, 0, 0, 1]])

def R(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, -s],
                     [s, c]])


if __name__ == '__main__':
    m = Problem()

    for _ in range(1):
        def arch_half():
            global dofs, p
            p = Problem()
            N = 16
            # n = int(np.floor(N/2))

            dofs = 2 * (3 * N - 2,)
            start = np.pi - np.arctan(600 / 800)
            end = np.pi / 2

            node_angles = np.linspace(start, end, N)
            node_points = 1000 * np.array([np.cos(node_angles), np.sin(node_angles)]).T
            for r1, r2 in zip(node_points, node_points[1:]):
                p.create_beam(r1, r2)

            p.constrained_dofs = np.array([0, 1, 3 * N - 3, 3 * N - 1])
            p.loads = np.array([*np.zeros(3 * N - 5), -400])

        def lin():
            global p, dofs
            p = Problem()
            dofs = (32, 32)
            p.create_beams((0,0),(1000,0), E=2.1e5, A=10, I=10**3/12, n=10)  # Aksial ok!
            p.fix(p.node_at((0,0)))
            p.load_node(p.node_at((1000,0)), (0, 0, 500000))

        def mt():
            global p, dofs
            p = Problem()
            dofs = (-2, -2)
            n = 1
            load = 15000
            p.create_beams((-500, 500), (500, 700), n=n)
            p.constrained_dofs = np.array([0, 1, 3 * n])
            p.loads = np.array([*np.zeros(3 * (n + 1) - 5), -load, 0])
            p.displacements = np.zeros(len(p.free_dofs()))

        def mt_spring():
            global p, dofs
            p = Problem()
            dofs = (7, 7)
            load = 4000
            p.create_beam((-500, 500), (500, 700))
            p.create_beam((500, 1200), (500, 700), E=210000 / 250)
            p.constrained_dofs = np.array([0, 1, 3, 6, 8])
            p.loads = np.array([0, 0, 0, -load])
            p.displacements = np.zeros(len(p.free_dofs()))

        def deg270():
            global p, dofs
            p = Problem()
            N = 31
            dofs = (46, 46)
            start = np.deg2rad(225)
            end = np.deg2rad(-45)
            node_angles = np.linspace(start, end, N)
            node_points = 500 * np.array([np.cos(node_angles), np.sin(node_angles)]).T + [0, 500]
            for r1, r2 in zip(node_points, node_points[1:]):
                p.create_beam(r1, r2)
            p.constrained_dofs = np.array([0, 1, 3 * N - 3, 3 * N - 2, 3 * N - 1])
            p.loads = np.array([0, *np.zeros(int((3 * N - 9) / 2)), 0, -300, 0, *np.zeros(int((3 * N - 9) / 2))])
            p.displacements = np.zeros(len(p.free_dofs()))

        def buckl():
            global p, dofs
            p = Problem()
            dofs = (60, 60)
            p.create_beams((0, 0), (1000, 0), n=20, I=1000)
            p.constrained_dofs = np.array([0, 1, 2, 61])
            p.loads = np.array([*np.zeros(len(p.free_dofs()) - 2), -80000000, 100000000])
            p.displacements = np.zeros(len(p.free_dofs()))

    lin()
    p.solve_newton()
    p.animate()