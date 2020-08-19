import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from extras import DSSCanvas


np.set_printoptions(suppress=True)


class Problem:

    def __init__(self):
        self.nodes = list()
        self.beams = list()

        self.constrained_dofs = []
        self.loads = np.array([])  # Joint loads
        self.member_loads = np.array([])  # Member loads, for distr loads and such
        self.forces = None  # Forces (at all nodes, incl removed dofs)
        self.displacements = None  # Assigned at solution ( self.solve() )
        self.solved = False

        # For incremental analysis
        self.incremental_loads = None
        self.incremental_displacements = None

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

    def beam_at(self, r1, r2):
        r1, r2 = np.asarray(r1), np.asarray(r2)
        for beam in self.beams:
            if beam == Beam(r1, r2) or beam == Beam(r2, r1):
                return self.beams.index(beam)

        print('No beam at', r1, r2)

    def reassign_dofs(self):
        self.loads = np.array([])
        for i,node in enumerate(self.nodes):
            node.number = i
            node.dofs = (3*i, 3*i+1, 3*i+2)
            self.loads = np.append(self.loads, node.loads)

    def reform_geometry(self):
        for element in self.beams:
            new_r1 = element.r1 + element.displacements[0:2]
            new_r2 = element.r2 + element.displacements[3:5]

            deformed_angle = np.arccos((np.dot(new_r2-new_r1, np.array([1, 0])) /
                                        np.linalg.norm(new_r2-new_r1)))
            element.k = beta(deformed_angle).T @ element.ke @ beta(deformed_angle)
            #print('Deformation angle', deformed_angle)

    def upd_obj_displacements(self):
        for node in self.nodes:
            node.displacements = self.displacements[np.array(node.dofs)]
        for beam in self.beams:
            beam.displacements = np.hstack((beam.nodes[0].displacements,
                                            beam.nodes[1].displacements))
            beam.forces = beam.k @ beam.displacements + beam.member_loads
            beam.stress = beam.cpl @ beam.beta(beam.angle) @ beam.forces
            #beam.nodes[0].forces = beam.forces[0:3]
            #beam.nodes[1].forces = beam.forces[3:6]  # No sign convention

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
            if np.all([type(element)==Rod for element in node.beams])\
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
            model_size = np.sqrt( (np.max(xy[:,0]) - np.min(xy[:,0]))**2 + (np.max(xy[:,1]) - np.min(xy[:,1]))**2)
            return model_size

    def K(self, reduced=False):
        self.remove_dofs()
        dofs = 3 * len(self.nodes)  # int: Number of system dofs
        K = sum(beam.Ki(dofs) for beam in self.beams)
        if not reduced:
            return K
        else:
            K = np.delete(K, self.constrained_dofs, axis=0)
            K = np.delete(K, self.constrained_dofs, axis=1)
            return K

    def Qf(self):  # Member force vector for distr loads
        dofs = 3*len(self.nodes)
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
        #self.ext_forces = self.K() @ self.displacements

        self.upd_obj_displacements()

        print('Nodal displacements', self.displacements)
        print('Nodal loads', self.loads)
        print('Member loads', self.member_loads)

        self.forces = np.array([beam.forces for beam in self.beams])
        # forces.shape == (n, 6), n: no. of beams

        self.solved = True
        return dr

    def solve_nlgeom(self, k):
        self.reassign_dofs()
        self.remove_dofs()
        free_dofs = np.delete(np.arange(3 * len(self.nodes)), self.constrained_dofs)

        self.displacements = np.zeros(3 * len(self.nodes))

        self.incremental_loads = np.zeros_like(self.loads) * np.zeros(k).reshape((k,1))
        self.incremental_displacements = np.zeros_like(self.loads) * np.zeros(k).reshape((k,1))
        self.repr_residual = []
        self.track_elm = []
        keepgoing = True

        for n in range(1,k+1):
            self.reform_geometry()  # Stiffness matrix self.K() is based on geometry after load step n-1
            displ = self.displacements[free_dofs]
            Kr = self.K(reduced=True)

            ext_force = n/k * self.loads[free_dofs]
            int_force = Kr @ displ

            ddispl = np.linalg.inv(Kr) @ (ext_force - int_force)
            displ = displ + ddispl
            self.displacements[free_dofs] = displ
            self.upd_obj_displacements()
            self.track_elm.append(self.beams[3].displacements)

            i = 1
            self.repr_residual.append(np.linalg.norm(ext_force - int_force))
            while not np.allclose(ext_force, int_force, rtol=0.02, atol=1):
                #print('Residual', residual)
                #self.reform_geometry()   # K = K(n, i-1)
                Kr = self.K(reduced=True)

                int_force = Kr @ self.displacements[free_dofs]
                ddispl = - np.linalg.inv(Kr) @ (ext_force - int_force)
                self.displacements[free_dofs] = self.displacements[free_dofs] + ddispl

                #self.upd_obj_displacements()

                i += 1
                maxcorrectors = 100

                if i%3 == 0 and i < 25:
                    #print('Load step {}, residual {}'.format(i, (ext_force-int_force)))
                    pass
                if i > maxcorrectors:
                    print('Step {} failed to converge after {} corrector steps'.format(n, maxcorrectors))
                    keepgoing = False
                    break
            if not keepgoing:
                self.incremental_displacements = self.incremental_displacements[0:n]
                self.incremental_loads = self.incremental_loads[0:n]
                break

            self.incremental_displacements[n-1] = self.displacements
            self.incremental_loads[n-1, free_dofs] = ext_force
            print('Load step {} converged after {} corrector step(s)'.format(n, i-1))
            #print('Displacements', self.displacements, 'Loads', ext_force)
            #print('Int force', int_force)
            #print('Element angle', )
            # The incr_displ[n-1] is still not equal to the actual displacement of the apex node

    def solve_nlgeom_energy(self, k):
        self.reassign_dofs()
        self.remove_dofs()
        free_dofs = np.delete(np.arange(3 * len(self.nodes)), self.constrained_dofs)

        self.displacements = np.zeros(3 * len(self.nodes))

        #loadstep = self.loads.copy() / k
        self.incremental_loads = np.zeros_like(self.loads) * np.zeros(k).reshape((k,1))
        self.incremental_displacements = np.zeros_like(self.loads) * np.zeros(k).reshape((k,1))


        for n in range(1,k+1):
            self.reform_geometry()  # Stiffness matrix self.K() is based on geometry after load step n-1
            displ = self.displacements[free_dofs]
            Kr = self.K(reduced=True)

            ext_force = n/k * self.loads[free_dofs]
            int_force = self.strain_energy_gradient()[free_dofs]
            residual = ext_force - int_force

            ddispl = np.linalg.inv(Kr) @ residual
            displ = displ + ddispl
            self.displacements[free_dofs] = displ
            self.upd_obj_displacements()

            i = 1
            while not np.allclose(int_force, self.strain_energy_gradient(), rtol=0.01, atol=0.1):
                #print('Residual', residual)

                self.reform_geometry()   # K = K(n, i-1)
                Kr = self.K(reduced=True)

                int_force = self.strain_energy_gradient()[free_dofs]
                ddispl = np.linalg.inv(Kr) @ (ext_force - int_force)
                self.displacements[free_dofs] = self.displacements[free_dofs] + ddispl
                self.upd_obj_displacements()
                #print('Load step {} corrector step {}'.format(n, i))
                #print('Displacements', self.displacements)
                #print('Loads int/ext', int_force, ext_force)
                i += 1


            self.incremental_displacements[n-1] = self.displacements
            self.incremental_loads[n-1, free_dofs] = ext_force
            print('Load step {} converged after {} corrector step(s)'.format(n, i))
            print('Displacements', self.displacements, 'Loads', ext_force)
            #print('Element angle', )
            # The incr_displ[n-1] is still not equal to the actual displacement of the apex node

    def solve_nlgeom_3(self, j):
        self.reassign_dofs()
        self.remove_dofs()
        free_dofs = np.delete(np.arange(3 * len(self.nodes)), self.constrained_dofs)

        self.displacements = np.zeros(3 * len(self.nodes))

        self.incremental_loads = np.zeros_like(self.loads) * np.zeros(j).reshape((j, 1))
        self.incremental_displacements = np.zeros_like(self.loads) * np.zeros(j).reshape((j, 1))
        self.repr_residual = []
        self.parameter_tracker = []
        ddispl = np.zeros_like(self.displacements[free_dofs])
        lbd = 0
        dlambda = 1
        keepgoing = True

        #for n in range(1, j + 1):
        while lbd <= j:  # A: Control parameter
            self.reform_geometry()  # Stiffness matrix self.K() is now based on geometry at step n-1

            q = (self.loads * dlambda / j)[free_dofs] - self.K(reduced=True) @ ddispl / dlambda  # ddispl from previous load step
            f = (self.loads * lbd / j)[free_dofs]

            displ = self.displacements[free_dofs]
            Kr = self.K(reduced=True)

            v = np.linalg.inv(Kr) @ q
            ddispl = v * dlambda

            self.displacements[free_dofs] = displ + ddispl
            self.upd_obj_displacements()

            residual = self.K(True)@self.displacements[free_dofs] - f
            #print(np.linalg.norm(ddispl))
            k = 0
            print(lbd)
            while not np.allclose(residual, 0, rtol=0.02, atol=0.1):
            #for s in range(5):
                #print(residual)
                #print('CORRECTOR LBD', lbd)

                c = np.dot(ddispl, ddispl)/self.model_size()**2 - dlambda**2    # Scalar constraint condition
                g = 0  # dc / d(lbd)
                aT = 2 * ddispl / self.model_size()**2  # dc / d(displacements)
                x,y = Kr.shape
                K_aug = np.empty((x+1, y+1))
                K_aug[0:x, 0:y] = Kr
                K_aug[0:x, x] = -q
                K_aug[x, 0:y] = aT
                K_aug[x,y] = g
                self.Kaug = K_aug

                r_aug = np.array([*residual, c])
                d_aug = np.linalg.inv(K_aug) @ (- r_aug)
                ddispl = d_aug[:-1]
                eta = d_aug[-1]  # Change in control param lbd for iterations
                #print(eta)

                self.displacements[free_dofs] = self.displacements[free_dofs] + eta*ddispl

                f = (self.loads * lbd / j)[free_dofs]
                residual = self.K(True) @ self.displacements[free_dofs] - f
                lbd += eta
                self.parameter_tracker.append(lbd)
                self.reform_geometry()
                self.upd_obj_displacements()
                k += 1
                if k > 15:
                    keepoing = False
                    break
            if not keepgoing:
                break

            try:
                self.incremental_displacements[int(np.round(lbd)) - 1] = self.displacements
                self.incremental_loads[int(np.round(lbd)) - 1, free_dofs] = f
            except:
                pass
            lbd += dlambda
            #print('Load step {} converged after {} corrector step(s)'.format(n, i - 1))
            # print('Displacements', self.displacements, 'Loads', ext_force)
            # print('Int force', int_force)
            # print('Element angle', )
            # The incr_displ[n-1] is still not equal to the actual displacement of the apex node

    def strain_energy(self, delta_displacements = None):
        if delta_displacements is None:
            delta_displacements = np.zeros_like(self.displacements)
            # OBS! self.displacements is initialized as None, and created at solution start

        return sum(element.strain_energy(delta_displacements[element.dofs]) for element in self.beams)

    def strain_energy_gradient(self, dui=1e-10):
        j = len(self.displacements)
        p = np.zeros(j)
        for k in range(j):
            delta_disp = np.zeros(j)
            delta_disp[k] = dui
            p[k] = (self.strain_energy(delta_disp) - self.strain_energy()) / dui
        return p  # p =~ self.K() @ self.displacements hopefully

    def work_energy(self):
        return np.dot(self.loads, self.displacements)/2
    
    def internal_forces(self, reduced=True):
        f_int = np.zeros(3*len(self.nodes))
        for beam in self.beams:
            f_beam = beam.Ex(3*len(self.nodes)) @ beam.k @ beam.displacements
            f_int = f_int + f_beam
        if reduced:
            free_dofs = np.delete(np.arange(3 * len(self.nodes)), self.constrained_dofs)
            return f_int[free_dofs]
        else:
            return f_int

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

    def animate(self, interval, startstep=0):
        fig = plt.figure()
        ax = fig.gca()

        id = self.incremental_displacements.reshape((-1, len(self.nodes), 3))[:,:,0:2]
        il = self.incremental_loads.reshape((-1, len(self.nodes), 3))
        # id[loadstep, node_id] = [u, v]

        ax.set_xlim(-self.model_size()*1.1, self.model_size()*1.1)
        ax.set_ylim(-self.model_size()*1.1, self.model_size()*1.1)
        
        line, = ax.plot([],[])
        line.set_data(self.nodal_coordinates.T)
        txt = ax.text(-self.model_size(), -self.model_size(), 'Load step')
        txt.set_text('Load step {} load norm {}')

        def animate_(k):
            if k < startstep - 10:
                k = startstep
            line.set_data(self.nodal_coordinates.T + id[k].T)
            txt.set_text('Load step {} load norm {}'.format(k, int(np.linalg.norm(il[k]))))


        self.anim = anm.FuncAnimation(fig, animate_,
                                 frames=id.shape[0], interval=interval, blit=False)

        #Writer = anm.writers['ffmpeg']
        #writer = Writer(fps=8, metadata=dict(artist='maglun engineering'),
        #                bitrate=1800)
        #self.anim.save('beam.mp4')
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
        self.loads = np.array([0,0,0]) # self.loads (global Fx, Fy, M) assigned on loading
        #self.forces = np.array([0,0,0])  # self.forces (Fx, Fy, M) calculated by self.abs_force

        self.number = None  # (node id) assigned on creation
        self.dofs = None  # self.dofs (dof1, dof2, dof3) assigned on creation
        self.displacements = np.array([0,0,0])  # self.displacement (d1, d2, d3) asssigned on solution
        self.boundary_condition = None  # 'fixed', 'pinned', 'roller', 'locked', 'glider'

        self.draw = draw  # Node is not drawn unless it is interesting

    def add_beam(self, beam):
        self.beams.append(beam)

    @property
    def abs_forces(self):
        self.forces = np.array([0,0,0])
        for element in self.beams:
            if np.array_equal(element.r1, self.r):
                self.forces = self.forces + np.abs(element.forces[0:3]) / 2
            elif np.array_equal(element.r2, self.r):
                self.forces = self.forces + np.abs(element.forces[3:6]) / 2
        return self.forces + np.abs(self.loads/2)

    def connected_nodes(self):
        other_nodes = []
        for beam in self.beams:
            for node in beam.nodes:
                if not node == self:
                    other_nodes.append(node)
        return other_nodes

    def translate(self, dx, dy):
        self.x, self.y = self.x+dx, self.y+dy
        self.r = np.array([self.x, self.y])

    def draw_on_canvas(self, canvas:DSSCanvas, **kwargs):
        canvas.draw_node(self.r, 2.5, **kwargs)

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
        self.z = z if z else np.sqrt(I/A)/3

        self.r1, self.r2 = np.asarray(r1), np.asarray(r2)
        self.angle = np.arctan2( *(self.r2 - self.r1)[::-1] )
        self.length = np.sqrt( np.dot(self.r2 - self.r1, self.r2 - self.r1) )

        self.number = None  # (beam id) assigned on creation
        self.displacements = np.zeros(6)  # global csys (1x6) assigned on solution
        self.forces = np.zeros(6)  # global csys (1x6) assigned on solution
        self.stress = np.zeros(6)  # local csys(1x6) assigned on solution
        # Stress given as sigma_x (top), sigma_x (bottom), tau_xy (average!)
        self.member_loads = np.zeros(6)  # local csys distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12
        self.distributed_load = 0
        # Distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12


        self.kn = A*E/self.length * (E*I/self.length**3)**(-1)
        self.ke = E*I/self.length**3 * np.array(
             [[self.kn, 0, 0, -self.kn, 0, 0],
              [0, 12, 6*self.length, 0, -12, 6*self.length],
              [0, 6*self.length, 4*self.length**2, 0, -6*self.length, 2*self.length**2],
              [-self.kn, 0, 0, self.kn, 0, 0],
              [0, -12, -6*self.length, 0, 12, -6*self.length],
              [0, 6*self.length, 2*self.length**2, 0, -6*self.length, 4*self.length**2]])

        self.k = self.beta(self.angle).T @ self.ke @ self.beta(self.angle)

        self.cpl = np.zeros((6,6))
        self.cpl_ = np.array([[1/self.A, 0, -self.z/self.I],
                             [1/self.A, 0, self.z/self.I],
                             [0,     1/self.A,   0]])
        self.cpl[0:3, 0:3] = self.cpl[3:6, 3:6] = self.cpl_

    def beta(self, angle):
        s, c = np.sin(angle), np.cos(angle)
        return np.array([[c, s, 0, 0, 0, 0],
                         [-s, c, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, c, s, 0],
                         [0, 0, 0, -s, c, 0],
                         [0, 0, 0, 0, 0, 1]])

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

    def Ex(self, newdim):
        dofs = self.nodes[0].dofs + self.nodes[1].dofs
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E

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
        N1,V1,M1,N2,V2,M2 = self.ke @ beta(self.angle) @ (self.displacements + delta_displacements)
        L = self.length
        return 1/(2*self.E*self.I) * (M1**2*L - M1*V1*L**2 + V1**2 * L**3/3) + \
               (N2**2 * L)/(2*self.E*self.A)

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

        self.kn = A*E/self.length
        self.ke = np.array(      [[self.kn, 0, 0, -self.kn, 0, 0],
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
            dofs = (35, 35)
            p.create_beams((-400, 500), (600, 500), n=11)  # Aksial ok!
            # p.create_beams((0,100),(0,700), n=5)  # ikke ok!
            # p.create_beams((-250,250),(250,750), n=11)  # ok!
            # Alle er dog ustabile ved store laster! Kun korrektorsteg gir ustabilitet
            # Kan lese av displacement history at sideveis forskyvning oscillerer. Korrektor forverrer?
            # Når bjelker roterer til å være vannrett opp/ned (alpha = pi/2), feiler løsningen
            p.constrained_dofs = np.array([0, 1, 2])
            p.loads = np.array([*np.zeros(3 * len(p.nodes) - len(p.constrained_dofs) - 3), 0, 0, 500000])
            p.displacements = np.zeros(len(p.free_dofs()))


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



    try:
        raise ValueError
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

        ax1.plot(-m.incremental_displacements[:,intr_displ], -m.incremental_loads[:,intr_load])
        ax2.plot(m.repr_residual)
        #"""
        ax1.set_xlabel('Displacement')
        ax1.set_ylabel('Load')
        ax2.set_xlabel('Control parameter')
        ax2.set_ylabel('Residual norm before iteration')
        #"""
        plt.tight_layout()
        plt.show()
    except:
        print('Pass')
        pass

