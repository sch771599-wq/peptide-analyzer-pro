import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import base64
import io

st.set_page_config(page_title="Ramachandran Analyzer", layout="wide")

# --- МАТЕМАТИКА ---
def calculate_dihedral(p1, p2, p3, p4):
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return round(float(np.degrees(np.arctan2(y, x))), 1)

# --- ПАРСИНГ ---
def parse_mol2_peptide(file_content):
    atom_data = [] 
    lines = file_content.splitlines()
    is_atom_block = False
    for line in lines:
        if "@<TRIPOS>ATOM" in line:
            is_atom_block = True
            continue
        if "@<TRIPOS>" in line and is_atom_block:
            is_atom_block = False
            break
        if is_atom_block:
            parts = line.split()
            if len(parts) > 1:
                element = "".join(filter(str.isalpha, parts[1]))[:1]
                atom_id = parts[0]
                atom_data.append({"id": atom_id, "display": f"{element}{atom_id}"})
    
    raw_mols = []
    blocks = file_content.split('@<TRIPOS>MOLECULE')
    for b in blocks:
        if b.strip():
            m = Chem.MolFromMol2Block('@<TRIPOS>MOLECULE' + b, removeHs=False)
            if m: raw_mols.append(m)
    
    if not raw_mols: return None, [], []
    mol_to_draw = Chem.RemoveHs(raw_mols[0], updateExplicitCount=True)
    rdDepictor.Compute2DCoords(mol_to_draw)
    
    list_labels = [a["display"] for a in atom_data if any(m.GetAtomWithIdx(i).GetSymbol() != 'H' for i in range(m.GetNumAtoms()) if atom_data[i]["display"] == a["display"])]
    
    h_idx = 0
    for i, atom in enumerate(raw_mols[0].GetAtoms()):
        if atom.GetSymbol() != 'H':
            if h_idx < mol_to_draw.GetNumAtoms():
                mol_to_draw.GetAtomWithIdx(h_idx).SetProp("atomNote", atom_data[i]["id"])
                h_idx += 1
            
    all_coords = []
    for m in raw_mols:
        conf = m.GetConformer()
        coords = {atom_data[j]["display"]: np.array(conf.GetAtomPosition(j)) for j in range(m.GetNumAtoms())}
        all_coords.append(coords)
        
    return mol_to_draw, all_coords, [a["display"] for a in atom_data if "H" not in a["display"]]

def render_svg(mol):
    d2d = rdMolDraw2D.MolDraw2DSVG(900, 400)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    return f'<div style="background: white; padding: 10px; border: 1px solid #ddd; border-radius: 10px;"><img src="data:image/svg+xml;base64,{b64}" width="100%"/></div>'

# --- ИНТЕРФЕЙС ---
st.title("🧬 Анализатор углов Phi/Psi")

uploaded_file = st.file_uploader("Загрузите .mol2 файл", type=["mol2"])

if uploaded_file:
    content = uploaded_file.getvalue().decode("utf-8")
    mol, confs, label_options = parse_mol2_peptide(content)
    
    if mol:
        st.write("### 🔬 Схема (номера атомов)")
        st.write(render_svg(mol), unsafe_allow_html=True)
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            phi_atoms = st.multiselect("Атомы для Phi (φ)", options=label_options, max_selections=4)
        with c2:
            psi_atoms = st.multiselect("Атомы для Psi (ψ)", options=label_options, max_selections=4)
            
        if len(phi_atoms) == 4 and len(psi_atoms) == 4:
            results = []
            for idx, c in enumerate(confs):
                results.append({
                    "Конформер": idx + 1,
                    "Phi": calculate_dihedral(*[c[at] for at in phi_atoms]),
                    "Psi": calculate_dihedral(*[c[at] for at in psi_atoms])
                })
            df = pd.DataFrame(results)
            
            st.markdown("---")
            # Основной контейнер для результатов
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.write("### График Рамачандрана")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["Phi"], y=df["Psi"], mode='markers+text',
                    text=df["Конформер"], textposition="top center",
                    marker=dict(size=10, color='#1f77b4', line=dict(width=1, color='white'))
                ))
                fig.update_layout(
                    width=500, height=500, template="plotly_white",
                    xaxis=dict(title="Phi (φ), градусы", range=[-180, 180], dtick=45, zeroline=True, zerolinewidth=2, zerolinecolor='black', showgrid=True, gridcolor='#eee'),
                    yaxis=dict(title="Psi (ψ), градусы", range=[-180, 180], dtick=45, zeroline=True, zerolinewidth=2, zerolinecolor='black', showgrid=True, gridcolor='#eee'),
                )
                st.plotly_chart(fig, use_container_width=False)
            
            with res_col2:
                st.write("### Таблица результатов")
                st.dataframe(df, hide_index=True, use_container_width=True)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                st.download_button("📥 Скачать Excel", buffer.getvalue(), "report.xlsx", use_container_width=True)