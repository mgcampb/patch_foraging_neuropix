import sys 
ibl_path = r'C:\Users\joshs\Documents\int-brain-lab'
sys.path.append(ibl_path)
import ibllib.atlas as atlas
import numpy as np

def get_parent_id_single_depth(id_allen,brain_id_single,depth): 
    """
    Get the parent id of an allen atlas brain region id up to a certain tree depth
    
    id_allen: allen tree dataframe indexed by id i.e. allen = pd.read_csv(allen_structure_tree_path).set_index('id')
    """
    parent_id = id_allen.loc[brain_id_single]['parent_structure_id'].astype(int)
    parent_depth = id_allen.loc[parent_id]['depth'].astype(int)
    if parent_depth > depth:
        parent_id = get_parent_id_single_depth(id_allen,parent_id,depth)
    return parent_id

def get_parent_id_depth(id_allen,brain_id,depth): 
    parent_id = np.nan + np.zeros(brain_id.shape)
    for i,this_brain_id in enumerate(brain_id): 
        parent_id[i] = get_parent_id_single_depth(id_allen,this_brain_id,depth)    
    return parent_id

def get_parent_id_nGenerations(id_allen,brain_id,nGenerations): 
    """
    Get the parent id of an allen atlas brain region id up a certain number of generations
    
    id_allen: allen tree dataframe indexed by id i.e. allen = pd.read_csv(allen_structure_tree_path).set_index('id')
    """
    try: 
        parent_id = id_allen.loc[brain_id]['parent_structure_id'].astype(int).values
    except: 
        print(brain_id)
        return
    if nGenerations == 1: 
        return parent_id
    else: 
        return get_parent_id_nGenerations(id_allen,parent_id,nGenerations-1)

def get_nearest_boundary(xyz_coords, allen, extent=100, steps=8, nGenerations=None,depth = None,
                             brain_atlas=None):
        """
        Adapted from IBLLib/ephys_alignment 
        
        note: xyz coords is w.r.t. bregma IN METERS
        
        Finds distance to closest neighbouring brain region along trajectory. For each point in
        xyz_coords computes the plane passing through point and perpendicular to trajectory and
        finds all brain regions that lie in that plane up to a given distance extent from specified
        point. Additionally, if requested, computes distance between the parents of regions.
        :param xyz_coords: 3D coordinates of points along probe or track
        :type xyz_coords: np.array((n_points, 3)) n_points: no. of points
        :param allen: dataframe containing allen info. Loaded from allen_structure_tree in
        ibllib/atlas
        :type allen: pandas Dataframe
        :param extent: extent of plane in each direction from origin in (um)
        :type extent: float
        :param steps: no. of steps to discretise plane into
        :type steps: int
        :param parent: Whether to also compute nearest distance between parents of regions
        :type parent: bool
        :return nearest_bound: dict containing results
        :type nearest_bound: dict
        """
        if not brain_atlas:
            brain_atlas = atlas.AllenAtlas(25)

        vector = atlas.Insertion.from_track(xyz_coords, brain_atlas=brain_atlas).trajectory.vector
        nearest_bound = dict()
        nearest_bound['dist'] = np.zeros((xyz_coords.shape[0]))
        nearest_bound['id'] = np.zeros((xyz_coords.shape[0]))
        # nearest_bound['adj_id'] = np.zeros((xyz_coords.shape[0]))
        nearest_bound['col'] = []

        if nGenerations: 
            nearest_bound['parent_gen%i_dist'%nGenerations] = np.zeros((xyz_coords.shape[0]))
            nearest_bound['parent_gen%i_id'%nGenerations] = np.zeros((xyz_coords.shape[0]))
            nearest_bound['parent_gen%i_col'%nGenerations] = []
        if depth: 
            nearest_bound['parent_depth%i_dist'%depth] = np.zeros((xyz_coords.shape[0]))
            nearest_bound['parent_depth%i_id'%depth] = np.zeros((xyz_coords.shape[0]))
            nearest_bound['parent_depth%i_col'%depth] = []
        
        for iP, point in enumerate(xyz_coords):
            d = np.dot(vector, point)
            x_vals = np.r_[np.linspace(point[0] - extent / 1e6, point[0] + extent / 1e6, steps),
                           point[0]]
            y_vals = np.r_[np.linspace(point[1] - extent / 1e6, point[1] + extent / 1e6, steps),
                           point[1]]

            X, Y = np.meshgrid(x_vals, y_vals)
            Z = (d - vector[0] * X - vector[1] * Y) / vector[2]
            XYZ = np.c_[np.reshape(X, X.size), np.reshape(Y, Y.size), np.reshape(Z, Z.size)]
            dist = np.sqrt(np.sum((XYZ - point) ** 2, axis=1)) # distance from each XYZ lookup point

            try: # Performs a 3D lookup from real world coordinates to the volume labels and return the regions ids according to the mapping 
                brain_id = brain_atlas.regions.get(brain_atlas.get_labels(XYZ))['id'] # get brain_ids associated w/ each lookup point
            except Exception as err:
                print(err)
                continue

            dist_sorted = np.argsort(dist)
            brain_id_sorted = brain_id[dist_sorted]
            
            nearest_bound['id'][iP] = brain_id_sorted[0]
            nearest_bound['col'].append(allen['color_hex_triplet'][np.where(allen['id'] == brain_id_sorted[0])[0][0]])
            
            bound_idx = np.where(brain_id_sorted != brain_id_sorted[0])[0]
            if np.any(bound_idx):
                nearest_bound['dist'][iP] = dist[dist_sorted[bound_idx[0]]] * 1e6
                # nearest_bound['adj_id'][iP] = brain_id_sorted[bound_idx[0]]
            else:
                nearest_bound['dist'][iP] = np.max(dist) * 1e6
                # nearest_bound['adj_id'][iP] = brain_id_sorted[0]
            
            # to make sure we don't run into root
            root_id = allen[allen['name'] == 'root']['id'].item()
            allen.loc[allen['name'] == 'root','parent_structure_id'] = root_id
            id_allen = allen.set_index('id')
            
            if nGenerations: # CSompute the parents up a certain number of generations
                # brain_parent = np.array([allen['parent_structure_id'][np.where(allen['id'] == br)[0][0]] for br in brain_id_sorted])
                
                try: 
                    brain_parent = get_parent_id_nGenerations(id_allen,brain_id_sorted,nGenerations)
                except: 
                    print("Problem with: ",brain_id_sorted) 
                    return brain_id_sorted
                    break
                brain_parent[np.isnan(brain_parent)] = 0
                
                nearest_bound['parent_gen%i_id'%nGenerations][iP] = brain_parent[0]
                nearest_bound['parent_gen%i_col'%nGenerations].append(allen['color_hex_triplet'][np.where(allen['id'] == brain_parent[0])[0][0]])

                parent_idx = np.where(brain_parent != brain_parent[0])[0]
                if np.any(parent_idx):
                    nearest_bound['parent_gen%i_dist'%nGenerations][iP] = dist[dist_sorted[parent_idx[0]]] * 1e6
                else:
                    nearest_bound['parent_gen%i_dist'%nGenerations][iP] = np.max(dist) * 1e6
                    
            if depth: # Compute the parents up a certain tree depth
                try: 
                    brain_parent = get_parent_id_depth(id_allen,brain_id_sorted,depth)
                except: 
                    print("Problem with: ",brain_id_sorted) 
                    return brain_id_sorted
                    break
            
                brain_parent[np.isnan(brain_parent)] = 0
                nearest_bound['parent_depth%i_id'%depth][iP] = brain_parent[0]
                nearest_bound['parent_depth%i_col'%depth].append(allen['color_hex_triplet'][np.where(allen['id'] == brain_parent[0])[0][0]])

                parent_idx = np.where(brain_parent != brain_parent[0])[0]
                if np.any(parent_idx):
                    nearest_bound['parent_depth%i_dist'%depth][iP] = dist[dist_sorted[parent_idx[0]]] * 1e6
                else:
                    nearest_bound['parent_depth%i_dist'%depth][iP] = np.max(dist) * 1e6
        return nearest_bound