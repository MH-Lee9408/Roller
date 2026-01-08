import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. FEM Solver Logic (SI 단위계 적용: N, m, Pa)
# ==========================================
def solve_fem(L1_mm, D1_mm, L2_mm, D2_mm, load_q_Nm, E_GPa, nu, num_elems_per_section, beam_theory):
    """
    Stepped Roller 1D FEM Solver (SI Units Basis)
    
    Inputs:
        L1_mm, D1_mm, ... : Dimensions in millimeters (will be converted to meters)
        load_q_Nm : Distributed Load in N/m
        E_GPa : Elastic Modulus in GPa (will be converted to Pa)
    """
    
    # --- [1] 단위 변환 (Unit Conversion to SI) ---
    # 길이: mm -> m
    L1 = L1_mm / 1000.0
    D1 = D1_mm / 1000.0
    L2 = L2_mm / 1000.0
    D2 = D2_mm / 1000.0
    
    # 탄성계수: GPa -> Pa (N/m^2)
    E = E_GPa * 1e9 
    G = E / (2 * (1 + nu))
    
    # 하중: N/m (이미 SI 단위이므로 변환 없음)
    q_load = load_q_Nm 

    # --- [2] 단면 물성 계산 함수 (Input: meters) ---
    def get_section_props(D):
        R = D / 2.0
        Area = np.pi * R**2
        I = (np.pi * R**4) / 4.0
        # 원형 단면 전단보정계수 (Cowper formula)
        k = 6 * (1 + nu) / (7 + 6 * nu) 
        return Area, I, k

    # --- [3] 노드 및 메쉬 생성 (Meters) ---
    x_sec1 = np.linspace(0, L1, num_elems_per_section + 1)
    x_sec2 = np.linspace(L1, L1 + L2, num_elems_per_section + 1)
    x_sec3 = np.linspace(L1 + L2, 2*L1 + L2, num_elems_per_section + 1)
    
    nodes = np.concatenate([x_sec1, x_sec2[1:], x_sec3[1:]])
    num_nodes = len(nodes)
    num_elements = num_nodes - 1
    dof = 2 * num_nodes
    
    # 글로벌 행렬 초기화
    K_global = np.zeros((dof, dof))
    F_global = np.zeros(dof)
    
    element_results = []
    
    # --- [4] 강성 행렬 조립 ---
    for i in range(num_elements):
        x_start = nodes[i]
        x_end = nodes[i+1]
        L_elem = x_end - x_start
        x_center = (x_start + x_end) / 2.0
        
        # 단차 처리 (현재 위치에 따른 직경 선택)
        if x_center < L1:
            D_curr = D1
        elif x_center > (L1 + L2):
            D_curr = D1
        else:
            D_curr = D2
            
        Area, I, k_shear = get_section_props(D_curr)
        
        # 요소 강성 행렬 (k_elem)
        k_elem = np.zeros((4, 4))
        
        if beam_theory == "Euler-Bernoulli":
            coeff = (E * I) / (L_elem**3)
            # 순서: [v1, th1, v2, th2]
            k_elem = coeff * np.array([
                [12, 6*L_elem, -12, 6*L_elem],
                [6*L_elem, 4*L_elem**2, -6*L_elem, 2*L_elem**2],
                [-12, -6*L_elem, 12, -6*L_elem],
                [6*L_elem, 2*L_elem**2, -6*L_elem, 4*L_elem**2]
            ])
        else: # Timoshenko
            Phi = (12 * E * I) / (k_shear * G * Area * L_elem**2)
            coeff = (E * I) / ((1 + Phi) * L_elem**3)
            k11 = 12
            k12 = 6 * L_elem
            k22 = (4 + Phi) * L_elem**2
            k22_neg = (2 - Phi) * L_elem**2
            
            k_elem = coeff * np.array([
                [k11, k12, -k11, k12],
                [k12, k22, -k12, k22_neg],
                [-k11, -k12, k11, -k12],
                [k12, k22_neg, -k12, k22]
            ])

        # 분포 하중 벡터 계산 (N/m 적용)
        f_elem = np.zeros(4)
        # 중앙부(L2) 구간 판별
        if L1 <= x_center <= (L1 + L2):
            # 하향 하중 (-)
            f_elem = np.array([
                -q_load * L_elem / 2,
                -q_load * L_elem**2 / 12,
                -q_load * L_elem / 2,
                 q_load * L_elem**2 / 12
            ])
            
        # Global Assembly
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for r in range(4):
            F_global[idx[r]] += f_elem[r]
            for c in range(4):
                K_global[idx[r], idx[c]] += k_elem[r, c]
        
        element_results.append({'k_elem': k_elem, 'dof_idx': idx, 'f_dist': f_elem})

    # --- [5] 경계 조건 및 풀이 ---
    bc_indices = [0, 2*(num_nodes-1)] # 양 끝단 v=0
    free_dof = [i for i in range(dof) if i not in bc_indices]
    
    K_reduced = K_global[np.ix_(free_dof, free_dof)]
    F_reduced = F_global[free_dof]
    
    U_reduced = np.linalg.solve(K_reduced, F_reduced)
    
    U_global = np.zeros(dof)
    U_global[free_dof] = U_reduced
    
    displacements = U_global[0::2] # Unit: meters
    
    # --- [6] SFD / BMD 후처리 (SI Units: N, N·m) ---
    shear_forces = []
    bending_moments = []
    
    for elem in element_results:
        u_elem = U_global[elem['dof_idx']]
        # {F} = [K]{u} - {F_load}
        f_int = np.dot(elem['k_elem'], u_elem) - elem['f_dist']
        
        V_left = f_int[0]
        M_left = f_int[1]
        
        shear_forces.append(V_left)      
        bending_moments.append(-M_left) # Beam convention adjustments
        
    shear_forces.append(shear_forces[-1])
    bending_moments.append(bending_moments[-1])
    
    return nodes, displacements, np.array(shear_forces), np.array(bending_moments)

# ==========================================
# 2. Streamlit Web UI Layout
# ==========================================
st.set_page_config(layout="wide", page_title="Roller FEM (SI Units)")

st.title("🎢 Stepped Roller FEM Analysis (SI Units)")
st.markdown("""
<style>
    .big-font { font-size:16px !important; color: #333; }
</style>
<div class='big-font'>
    <b>단위 시스템(Unit System):</b> 모든 계산은 <b>SI 단위(Meters, Newtons, Pascals)</b>로 수행됩니다.<br>
    입력은 편의상 mm, GPa를 사용하지만 내부는 m, Pa, N/m로 변환됩니다.
</div>
""", unsafe_allow_html=True)
st.divider()

# --- 입력 패널 ---
st.subheader("📝 해석 파라미터 입력")

col_input1, col_input2, col_input3 = st.columns(3)

with col_input1:
    st.markdown("##### 1. 기하 형상 (mm)")
    st.caption("※ 계산 시 미터(m)로 변환됨")
    input_L1 = st.number_input("좌측 지지부 길이 L1 (mm)", value=200.0)
    input_D1 = st.number_input("좌측 지지부 직경 D1 (mm)", value=30.0)
    input_L2 = st.number_input("중앙 롤러 길이 L2 (mm)", value=600.0)
    input_D2 = st.number_input("중앙 롤러 직경 D2 (mm)", value=80.0)

with col_input2:
    st.markdown("##### 2. 하중 및 물성 (SI)")
    # 기존 N/mm 입력을 N/m로 변경 (1 N/mm = 1000 N/m)
    # 기본값을 10000 N/m (즉 10 N/mm)로 설정하여 스케일 유지
    input_load = st.number_input("분포 하중 q (N/m)", value=10000.0, step=1000.0, help="1 N/mm = 1000 N/m 입니다.")
    input_E = st.number_input("탄성계수 E (GPa)", value=210.0)
    input_nu = st.number_input("포아송비", value=0.3)

with col_input3:
    st.markdown("##### 3. 해석 옵션")
    input_mesh = st.slider("구간당 요소 수", 2, 50, 20)
    input_theory = st.selectbox("빔 이론", ["Euler-Bernoulli", "Timoshenko"])
    # m 단위 변위는 매우 작으므로 확대 배율 조정 필요
    input_scale = st.number_input("변위 확대 배율", value=1000.0)

solve_btn = st.button("🚀 FEM 해석 실행 (Calculate)", type="primary", use_container_width=True)
st.divider()

# ==========================================
# 3. 결과 가시화 (Visualization with Correct Units)
# ==========================================
if solve_btn:
    with st.spinner('Calculating in SI Units...'):
        nodes_m, disp_m, sfd_N, bmd_Nm = solve_fem(
            input_L1, input_D1, input_L2, input_D2, 
            input_load, input_E, input_nu, 
            input_mesh, input_theory
        )
        
        # 시각화 편의를 위해 결과를 mm로 변환 (그래프 X축 등)
        nodes_mm = nodes_m * 1000.0
        disp_mm = disp_m * 1000.0 # 처짐량 mm 변환
        max_disp_mm = np.min(disp_mm)

    # --- 결과 요약 ---
    st.subheader("📊 해석 결과 리포트")
    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("최대 처짐 (Max Deflection)", f"{max_disp_mm:.4f} mm")
    col_res2.metric("최대 전단력 (Max Shear)", f"{np.max(np.abs(sfd_N)):.1f} N")
    col_res3.metric("최대 모멘트 (Max Moment)", f"{np.max(np.abs(bmd_Nm)):.1f} N·m")

    # --- 그래프 그리기 ---
    tab1, tab2 = st.tabs(["🖼️ 처짐 형상 (Deflection)", "📈 SFD & BMD"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        
        # 롤러 형상 (배경) - mm 단위로 그리기
        ax1.add_patch(patches.Rectangle((0, -input_D1/2), input_L1, input_D1, fc='lightgray', ec='black', alpha=0.5))
        ax1.add_patch(patches.Rectangle((input_L1, -input_D2/2), input_L2, input_D2, fc='gray', ec='black', alpha=0.5))
        ax1.add_patch(patches.Rectangle((input_L1+input_L2, -input_D1/2), input_L1, input_D1, fc='lightgray', ec='black', alpha=0.5))
        
        # 처짐 곡선 (mm 단위 + 확대)
        deformed_y_mm = disp_mm * input_scale
        ax1.plot(nodes_mm, deformed_y_mm, 'r-', linewidth=2, label=f'Deflection (x{input_scale})')
        
        ax1.set_title(f"Deformed Shape ({input_theory})")
        ax1.set_xlabel("Position (mm)")
        # Y축은 형상(mm) + 처짐(mm)
        ax1.set_ylabel("Diameter / Deflection (mm)")
        ax1.axis('equal')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        st.pyplot(fig1)

    with tab2:
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # SFD (Shear Force Diagram)
        ax2.plot(nodes_mm, sfd_N, 'b-', linewidth=2)
        ax2.fill_between(nodes_mm, sfd_N, 0, color='blue', alpha=0.1)
        ax2.set_ylabel("Shear Force (N)", fontsize=12, fontweight='bold') # 요청하신 단위
        ax2.set_title("Shear Force Diagram (SFD)")
        ax2.grid(True)
        
        # BMD (Bending Moment Diagram)
        ax3.plot(nodes_mm, bmd_Nm, 'g-', linewidth=2)
        ax3.fill_between(nodes_mm, bmd_Nm, 0, color='green', alpha=0.1)
        ax3.set_xlabel("Position (mm)")
        ax3.set_ylabel("Bending Moment (N·m)", fontsize=12, fontweight='bold') # 요청하신 단위
        ax3.set_title("Bending Moment Diagram (BMD)")
        ax3.grid(True)
        
        st.pyplot(fig2)

else:
    st.info("입력값을 확인하고 실행 버튼을 눌러주세요.")
