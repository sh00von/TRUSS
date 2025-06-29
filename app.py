import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from io import StringIO

app = Flask(__name__)

# Ensure the 'templates' folder exists for Flask to find the HTML file
if not os.path.exists('templates'):
    os.makedirs('templates')

@app.route('/')
def index():
    """Serves the main HTML page for the truss builder."""
    return render_template('index.html')

# --- Helper function for the solver ---
def solve_truss_logic(props, nodes, members, supports, loads, log_steps=False):
    """
    This is the core solver logic. It takes dictionaries of truss components 
    and returns the analysis results. If log_steps is True, returns a detailed log.
    """
    log = StringIO() if log_steps else None
    if not nodes: return {"error": "No nodes were defined."}, None, None, (log.getvalue() if log else None)
    node_ids = [n['id'] for n in nodes]
    if not node_ids: return {"error": "Node list is empty."}, None, None, (log.getvalue() if log else None)
    sorted_node_ids = sorted(node_ids)
    node_id_map = {node_id: i for i, node_id in enumerate(sorted_node_ids)}
    E = props.get('E', 200e9)
    A = props.get('A', 0.01)
    EA = E * A
    num_dofs = 2 * len(node_ids)
    K = np.zeros((num_dofs, num_dofs))
    F = np.zeros(num_dofs)
    nodes_dict = {n['id']: {'pos': (n['x'], n['y'])} for n in nodes}
    if log:
        log.write(f"Nodes: {nodes_dict}\n")
        log.write(f"Members: {members}\n")
        log.write(f"Supports: {supports}\n")
        log.write(f"Loads: {loads}\n")
        log.write(f"E = {E}, A = {A}, EA = {EA}\n")
    # For member compatibility with string IDs (e.g., 'M1'), allow both dict and list
    if isinstance(members, dict):
        member_list = []
        for m_id, m in members.items():
            # Accept both {'nodes': (n1, n2)} and {'startNode': n1, 'endNode': n2}
            if 'nodes' in m:
                member_list.append({'startNode': m['nodes'][0], 'endNode': m['nodes'][1]})
            else:
                member_list.append({'startNode': m['startNode'], 'endNode': m['endNode']})
        members = member_list
    for mem in members:
        n1_id, n2_id = mem['startNode'], mem['endNode']
        if n1_id not in nodes_dict or n2_id not in nodes_dict: continue
        n1_pos, n2_pos = nodes_dict[n1_id]['pos'], nodes_dict[n2_id]['pos']
        dx, dy = n2_pos[0] - n1_pos[0], n2_pos[1] - n1_pos[1]
        L = np.sqrt(dx**2 + dy**2)
        if L == 0: continue
        c, s = dx / L, dy / L
        k_e = (EA / L) * np.array([[c*c, c*s, -c*c, -c*s], [c*s, s*s, -c*s, -s*s], [-c*c, -c*s, c*c, c*s], [-c*s, -s*s, c*s, s*s]])
        idx1, idx2 = node_id_map[n1_id], node_id_map[n2_id]
        dof_map = [2*idx1, 2*idx1+1, 2*idx2, 2*idx2+1]
        for i in range(4):
            for j in range(4): K[dof_map[i], dof_map[j]] += k_e[i, j]
        if log:
            log.write(f"\nMember {n1_id}-{n2_id}: L={L:.3f}, c={c:.3f}, s={s:.3f}\n")
            log.write(f"Local stiffness matrix (k_e):\n{k_e}\n")
            log.write(f"DOF map: {dof_map}\n")
    for load in loads:
        if load['nodeId'] in node_id_map:
            idx = node_id_map[load['nodeId']]
            F[2 * idx] += load['fx']; F[2 * idx + 1] += load['fy']
    if log:
        log.write(f"\nGlobal stiffness matrix (K):\n{K}\n")
        log.write(f"Global force vector (F):\n{F}\n")
    restrained_dofs = []
    for support in supports:
        if support['nodeId'] in node_id_map:
            idx = node_id_map[support['nodeId']]
            if support['type'].upper() == 'PINNED': restrained_dofs.extend([2 * idx, 2 * idx + 1])
            elif support['type'].upper() == 'ROLLER': restrained_dofs.append(2 * idx + 1)
    K_original = K.copy()
    for dof in restrained_dofs:
        K[dof, :], K[:, dof] = 0, 0
        K[dof, dof], F[dof] = 1, 0
    if log:
        log.write(f"\nK after applying supports (boundary conditions):\n{K}\n")
        log.write(f"F after supports: {F}\n")
        log.write(f"Restrained DOFs: {restrained_dofs}\n")
    try:
        U = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        return {"error": "The truss is unstable. The stiffness matrix is singular. Check supports and connectivity."}, None, None, (log.getvalue() if log else None)
    R_vector = K_original @ U
    reactions = {s['nodeId']: {'rx': 0, 'ry': 0} for s in supports}
    for dof in restrained_dofs:
        matrix_idx = dof // 2
        original_node_id = sorted_node_ids[matrix_idx]
        if original_node_id in reactions:
            force_type = 'rx' if dof % 2 == 0 else 'ry'
            reactions[original_node_id][force_type] = R_vector[dof]
    member_forces = []
    for i, mem in enumerate(members):
        n1_id, n2_id = mem['startNode'], mem['endNode']
        if n1_id not in nodes_dict or n2_id not in nodes_dict: continue
        n1_pos, n2_pos = nodes_dict[n1_id]['pos'], nodes_dict[n2_id]['pos']
        dx, dy = n2_pos[0] - n1_pos[0], n2_pos[1] - n1_pos[1]
        L = np.sqrt(dx**2 + dy**2)
        if L == 0:
            member_forces.append({'memberIndex': i, 'force': 0})
            continue
        c, s = dx / L, dy / L
        idx1, idx2 = node_id_map[n1_id], node_id_map[n2_id]
        dof_map = [2*idx1, 2*idx1+1, 2*idx2, 2*idx2+1]
        force = (EA / L) * (np.array([-c, -s, c, s]) @ U[dof_map])
        member_forces.append({'memberIndex': i, 'force': force})
        if log:
            log.write(f"\nMember {n1_id}-{n2_id} force: {force:.3f}\n")
    if log:
        log.write(f"\nNodal displacements (U):\n{U}\n")
        log.write(f"\nReactions at supports:\n{reactions}\n")
        log.write(f"\nMember forces:\n{member_forces}\n")
    return None, member_forces, reactions, (log.getvalue() if log else None)


@app.route('/solve_truss', methods=['POST'])
def solve_truss_endpoint():
    """
    Receives truss data from the frontend, including material properties,
    solves it, and returns the results.
    """
    data = request.json
    props = data.get('properties', {"E": 200e9, "A": 0.01})
    error, member_forces, reactions, _ = solve_truss_logic(
        props, data['nodes'], data['members'], data['supports'], data['loads']
    )
    if error:
        return jsonify(error), 400
    return jsonify({
        'memberForces': member_forces,
        'reactions': reactions
    })

# --- New endpoint for detailed calculation log ---
@app.route('/truss_details', methods=['POST'])
def truss_details_endpoint():
    data = request.json
    props = data.get('properties', {"E": 200e9, "A": 0.01})
    _, _, _, log = solve_truss_logic(
        props, data['nodes'], data['members'], data['supports'], data['loads'], log_steps=True
    )
    return log or "No log available."

if __name__ == "__main__":
    app.run(debug=True)
