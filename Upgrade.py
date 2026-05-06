import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# 1. НАСТРОЙКИ СТРАНИЦЫ
st.set_page_config(page_title="Ramachandran Analyzer", layout="wide")

# 2. МАТЕМАТИЧЕСКИЙ МОДУЛЬ (ТВОЯ ОРИГИНАЛЬНАЯ ЛОГИКА)
def calculate_dihedral(p1, p2, p3, p4):
    """Точный расчет торсионного угла по 4 точкам."""
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return round(float(np.degrees(np.arctan2(y, x))), 1)

# 3. ПАРСИНГ MOL2 (ВОССТАНОВЛЕННАЯ НУМЕРАЦИЯ)
def parse_mol2_peptide(file_content):
    original_ids = []
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
            if len(parts) > 0:
                # ВОЗВРАЩАЕМ ЦИФРОВУЮ НУМЕРАЦИЮ (parts[0])
                original_ids.append(str(parts[0]))

    raw_mols = []
    blocks = file_content.split('@<TRIPOS>MOLECULE')
    for b in blocks:
        if b.strip():
            m = Chem.MolFromMol2Block('@<TRIPOS>MOLECULE' + b, removeHs=False)
            if m: raw_mols.append(m)
    
    if not raw_mols:
        return None, [], []

    mol_to_draw = Chem.RemoveHs(raw_mols[0])
    rdDepictor.Compute2DCoords(mol_to_draw)
    
    heavy_labels = []
    h_idx = 0
    for i, atom in enumerate(raw_mols[0].GetAtoms()):
        if atom.GetSymbol() != 'H':
            orig_id = original_ids[i]
            label = f"{atom.GetSymbol()}{orig_id}"
            heavy_labels.append(label)
            mol_to_draw.GetAtomWithIdx(h_idx).SetProp("mol2_id", orig_id)
            h_idx += 1

    all_coords = []
    for m in raw_mols:
        conf = m.GetConformer()
        coords = {f"{a.GetSymbol()}{original_ids[j]}": np.array(conf.GetAtomPosition(j)) 
                  for j, a in enumerate(m.GetAtoms())}
        all_coords.append(coords)
        
    return mol_to_draw, all_coords, heavy_labels

# 4. РЕНДЕРИНГ
def render_mol(mol):
    d2d = rdMolDraw2D.MolDraw2DSVG(1000, 350)
    opts = d2d.drawOptions()
    opts.addAtomIndices = False
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.HasProp("mol2_id"):
            atom.SetProp("atomNote", atom.GetProp("mol2_id"))
        opts.atomLabels[i] = "" 
    opts.annotationFontScale = 0.8
    opts.bondLineWidth = 2.5
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText().replace('<svg', '<svg width="100%"')

# 5. ИНТЕРФЕЙС
st.title("🧬 Анализатор Phi/Psi (MOL2)")

up = st.file_uploader("Загрузите .mol2 файл (ансамбль конформеров)", type=['mol2'])

if up:
    try:
        content = up.read().decode("utf-8")
        mol, confs, labels = parse_mol2_peptide(content)
        
        if mol:
            st.subheader("1. Структура и индексы атомов")
            st.components.v1.html(render_mol(mol), height=380)
            st.divider()
            
            st.subheader("2. Параметры расчета")
            c1, c2 = st.columns(2)
            with c1:
                phi_at = st.multiselect("Атомы для Phi (φ):", options=labels, key="p")
            with c2:
                psi_at = st.multiselect("Атомы для Psi (ψ):", options=labels, key="s")

            if len(phi_at) == 4 and len(psi_at) == 4:
                results = []
                for idx, c in enumerate(confs):
                    results.append({
                        "Конформер": idx + 1,
                        "Phi": calculate_dihedral(c[phi_at[0]], c[phi_at[1]], c[phi_at[2]], c[phi_at[3]]),
                        "Psi": calculate_dihedral(c[psi_at[0]], c[psi_at[1]], c[psi_at[2]], c[psi_at[3]])
                    })
                df = pd.DataFrame(results)
                
                # ФОРМАТИРОВАНИЕ ЧИСЕЛ (добавляем .0)
                df_disp = df.copy()
                df_disp["Phi"] = df_disp["Phi"].map(lambda x: f"{x:.1f}")
                df_disp["Psi"] = df_disp["Psi"].map(lambda x: f"{x:.1f}")

                col_tab, col_plot = st.columns([1, 1.5])
                
                with col_tab:
                    st.write("**Данные расчета:**")
                    st.dataframe(df_disp, hide_index=True, use_container_width=True)
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Ramachandran')
                    
                    st.download_button(
                        label="📥 Экспорт в Excel",
                        data=output.getvalue(),
                        file_name="phi_psi_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col_plot:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["Phi"], y=df["Psi"],
                        mode='markers+text',
                        text=df["Конформер"],
                        textposition="top center",
                        marker=dict(size=12, color='#1f77b4', opacity=0.8, line=dict(width=1, color='white'))
                    ))
                    
                    # ИСПРАВЛЕННЫЕ ОСИ (БЕЗ 360, ШАГ 45)
                    fig.update_xaxes(
                        title="Phi (φ), градусы",
                        range=[-180, 180],
                        tickmode='linear', tick0=-180, dtick=45,
                        constrain='domain', showgrid=True, gridcolor='#e0e0e0',
                        zeroline=True, zerolinecolor='#444', zerolinewidth=2,
                        mirror=True, showline=True, linecolor='black'
                    )
                    
                    fig.update_yaxes(
                        title="Psi (ψ), градусы",
                        range=[-180, 180],
                        tickmode='linear', tick0=-180, dtick=45,
                        scaleanchor="x", scaleratio=1,
                        showgrid=True, gridcolor='#e0e0e0',
                        zeroline=True, zerolinecolor='#444', zerolinewidth=2,
                        mirror=True, showline=True, linecolor='black'
                    )
                    
                    fig.update_layout(
                        template="plotly_white",
                        height=600, margin=dict(l=50, r=50, t=50, b=50),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка: {e}")