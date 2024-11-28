import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pandas as pd
import plotly.io as pio
import plotly.express as px

# Configuration de la page Streamlit
st.set_page_config(layout="wide", page_title="Simulateur Masse-Ressort Avancé")

# En-tête avec logo, informations sur l'université et les intervenants
col1, col2 = st.columns([1, 3])
with col1:
    st.image("logo_unchk.png", width=400)  # Ajouter le logo en haut à gauche

with col2:
    st.markdown(
"""
    ### Master 1 Calcul Scientifique et Modélisation | UE 4111 : Equations différentielles et algorithme sur les matricielles
    **Responsable de Formation** : Pr. Oumar Diop (oumar.diop@unchk.edu.sn)  
    **Tuteur** : Dr. Cheikh Gueye (cheikh1.gueye@unchk.sn)  
    **Étudiant** : Yaya Toure (yaya.toure@etu.unchk.sn)  
    [LinkedIn](https://www.linkedin.com/in/yaya-toure-8251a4280/) | [GitHub](https://github.com/YAYATOURE)
    """)

# Titre principal
st.title("Analyse Avancée du Système Masse-Ressort avec Étude de Convergence")

# Barre latérale pour les paramètres
with st.sidebar:
    st.header("Paramètres du système")
    
    # Paramètres physiques de base
    st.subheader("Paramètres physiques")
    mass = st.slider("Masse (kg)", 0.1, 5.0, 1.0, 0.1)
    k = st.slider("Constante de raideur (N/m)", 0.1, 10.0, 1.0, 0.1)
    
    # Calcul de omega_0
    omega_0 = np.sqrt(k / mass)
    st.info(f"Fréquence naturelle (ω₀) : {omega_0:.2f} rad/s")
    
    # Configuration des scénarios de frottement
    st.subheader("Scénarios de frottement")
    show_all_cases = st.checkbox("Afficher tous les cas de frottement", True)
    
    
    # Amortissement
    st.subheader("Amortissement")
    damping_ratio = st.slider("Ratio d'amortissement (ζ)", 0.0, 2.0, 0.5, 0.1)
    alpha = damping_ratio * omega_0
    
    if not show_all_cases:
        damping_ratio = st.slider("Ratio d'amortissement (ζ)", 0.0, 2.0, 0.5, 0.1)
        alpha = damping_ratio * omega_0
    
    # Conditions initiales
    st.subheader("Conditions initiales")
    x0 = st.slider("Position initiale (m)", -2.0, 2.0, 1.0, 0.1)
    v0 = st.slider("Vitesse initiale (m/s)", -2.0, 2.0, 0.0, 0.1)
    
    # Paramètres de simulation
    st.subheader("Paramètres de simulation")
    t_final = st.slider("Temps de simulation (s)", 1.0, 30.0, 20.0, 1.0)
    dt_values = [0.1, 0.05, 0.01, 0.005]
    dt = st.select_slider("Pas de temps (s)", options=dt_values, value=0.01)

# Fonction du système d'équations
def system(t, X, alpha):
    x, v = X
    dxdt = v
    dvdt = -omega_0**2 * x - 2 * alpha * v
    return np.array([dxdt, dvdt])

# Méthode de Runge-Kutta d'ordre 4
def runge_kutta_4(f, t0, tf, X0, dt, alpha):
    t_values = np.arange(t0, tf + dt/2., dt) # Adjusted to include tf in range.
    X_values = np.zeros((len(t_values), len(X0)))
    X_values[0] = X0
    
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        X = X_values[i - 1]
        k1 = f(t, X, alpha)
        k2 = f(t + dt/2., X + dt*k1/2., alpha)
        k3 = f(t + dt/2., X + dt*k2/2., alpha)
        k4 = f(t + dt , X + dt*k3 , alpha)
        X_values[i] = X + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t_values , X_values

# Fonction de calcul d'erreur de convergence
def convergence_error_adjusted(fine_solution , coarse_solution):
    fine_solution_adjusted = fine_solution[::2]
    return np.linalg.norm(fine_solution_adjusted - coarse_solution , ord=2) / len(coarse_solution)

# Création du graphique principal pour les oscillations
fig1 = go.Figure()

if show_all_cases:
    alpha_values = [1.5 * omega_0 , 0.5 * omega_0 , 0] # Inclure le cas critique

    # Ajout des scénarios de frottement initiaux avec les nouvelles légendes
    for alpha in alpha_values:
        t_values , X_values = runge_kutta_4(system , 0 , t_final , [x0 , v0] , dt , alpha)

        if alpha > omega_0: # Cas de frottements importants
            label = f"Frottements importants: α² > ω₀²\nα = {alpha:.2f}, ω₀ = {omega_0:.2f}"
        elif alpha < omega_0 and alpha > 0: # Cas de frottements faibles
            label = f"Frottements faibles: 0 < α² < ω₀²\nα = {alpha:.2f}, ω₀ = {omega_0:.2f}"
        else: # Cas sans frottements
            label = f"Sans frottements: α = 0\nω₀ = {omega_0:.2f}"

        fig1.add_trace(go.Scatter(x=t_values , y=X_values[:, 0] , mode='lines' , name=label))

# Cas critique 
alpha_critical = omega_0 
t_values_critical , X_values_critical = runge_kutta_4(system , 0 , t_final ,[x0 , v0] , dt , alpha_critical)
fig1.add_trace(go.Scatter(x=t_values_critical , y=X_values_critical[:, 0] , mode='lines',
                         name=f"Cas critique: α² = ω₀²\nα = {alpha_critical:.2f}, ω₀ = {omega_0:.2f}",
                         line=dict(dash='dash' , color='green')))

# Personnalisation du graphique principal 
fig1.update_layout(
    title="Oscillations du système masse-ressort avec différents niveaux de frottement",
    xaxis_title="Temps (s)",
    yaxis_title="Déplacement x(t)",
    template="plotly_dark"
)

# Affichage du graphique interactif pour les oscillations 
st.plotly_chart(fig1)

# Création des subplots pour les erreurs de convergence 
fig2 = make_subplots(rows=2 , cols=2,
                     subplot_titles=[f"Erreur pour α={alpha:.2f}" for alpha in alpha_values + [alpha_critical]],
                     shared_xaxes=True,
                     shared_yaxes=True)

# Calcul des erreurs de convergence et ajout aux subplots 
for idx , alpha in enumerate(alpha_values + [alpha_critical]):
    errors = []
    
    for dt_coarse in dt_values:
        _, coarse_solution   = runge_kutta_4(system , 0 , t_final ,[x0,v0] , dt_coarse ,alpha) 
        _, fine_solution     = runge_kutta_4(system , 0,t_final,[x0,v0],dt_coarse/2.,alpha) 
        error                = convergence_error_adjusted(fine_solution[:, 0], coarse_solution[:, 0])
        
        errors.append(error)

    row, col = divmod(idx, 2)
    
    fig2.add_trace(
        go.Scatter(x=dt_values, y=errors, mode='lines+markers', name=f"Erreur: α={alpha:.2f}"),
        row=row + 1,
        col=col + 1)

# Personnalisation des subplots pour les erreurs de convergence 
fig2.update_layout(
    title="Erreur de convergence pour différents scénarios",
    xaxis_title="Pas de temps (dt)",
    yaxis_title="Erreur de convergence",
    template="plotly_dark"
)

# Affichage des subplots interactifs pour les erreurs de convergence 
st.plotly_chart(fig2)

# Fonction de calcul d'erreur de convergence ajustée
def convergence_error_adjusted(fine_solution, coarse_solution):
    # Adapter la solution fine pour correspondre à la taille de la solution coarse
    fine_solution_adjusted = fine_solution[::2]  # On prend un point sur deux pour correspondre
    return np.linalg.norm(fine_solution_adjusted - coarse_solution, ord=2) / len(coarse_solution)

# Listes pour stocker les résultats
convergence_errors = []
execution_times = []
case_labels = []

# Définir une liste de valeurs pour dt
dt_values = [0.1, 0.05, 0.01, 0.005]  # Exemple de valeurs pour dt

# Calcul de convergence pour chaque cas et chaque pas de temps
for alpha in alpha_values + [alpha_critical]:
    # Détermination de l'étiquette pour chaque cas
    if alpha > omega_0:
        label = f"Frottements importants: α² > ω₀² (α = {alpha:.2f}, ω₀ = {omega_0:.2f})"
    elif alpha < omega_0 and alpha > 0:
        label = f"Frottements faibles: 0 < α² < ω₀² (α = {alpha:.2f}, ω₀ = {omega_0:.2f})"
    elif alpha == 0:
        label = f"Sans frottements: α = 0 (ω₀ = {omega_0:.2f})"
    else:
        label = f"Cas critique: α² = ω₀² (α = {alpha:.2f}, ω₀ = {omega_0:.2f})"

    # Calcul du temps d'exécution et erreur de convergence pour chaque pas de temps
    for dt in dt_values:
        # Exécution pour un pas de temps donné
        start_time = time.time()
        _, coarse_solution = runge_kutta_4(system, 0, t_final, [x0, v0], dt, alpha)
        execution_time = time.time() - start_time

        # Calcul de la solution fine avec un pas de temps divisé par 2
        _, fine_solution = runge_kutta_4(system, 0, t_final, [x0, v0], dt / 2, alpha)

        # Calcul de l'erreur de convergence avec ajustement
        error = convergence_error_adjusted(fine_solution[:, 0], coarse_solution[:, 0])

        # Stockage des résultats
        convergence_errors.append(error)
        execution_times.append(execution_time)
        case_labels.append(f"{label} (dt={dt})")

# Création du DataFrame pour afficher les résultats
results_df = pd.DataFrame({
    "Cas": case_labels,
    "Erreur de Convergence": convergence_errors,
    "Temps d'Exécution (s)": execution_times
})

# Formater l'erreur de convergence en notation scientifique
results_df["Erreur de Convergence"] = results_df["Erreur de Convergence"].apply(lambda x: f"{x:.2e}")

# Affichage des résultats et graphique côte à côte
st.subheader("Résultats de convergence et de performance")
col1, col2 = st.columns(2)

with col1:
    st.dataframe(results_df)

with col2:
    # Création du graphique des erreurs de convergence
    pio.templates.default = "plotly_dark"

    fig = px.bar(
        results_df,
        x='Cas',
        y='Erreur de Convergence',
        title='Erreur de convergence pour chaque scénario',
        labels={'Cas': 'Scénario de Simulation', 'Erreur de Convergence': 'Erreur de Convergence'}
    )

    # Personnalisation du graphique
    fig.update_layout(
        title=dict(
            text='Erreur de convergence pour chaque scénario',
            x=0.5,  # Centrer le titre
            xanchor='center'
        ),
        xaxis=dict(
            title='Scénario de Simulation',
            tickangle=45  # Rotation des étiquettes pour une meilleure lisibilité
        ),
        yaxis=dict(
            title='Erreur de Convergence'
        ),
        template="plotly_dark"
    )

    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig)


#----------------------
# Titre principal
st.title("Simulateur de Système Masse-Ressort avec Amortissement")


def system(t, X, alpha, omega_0):
    x, v = X
    dxdt = v
    dvdt = -omega_0**2 * x - 2 * alpha * v
    return np.array([dxdt, dvdt])

# Méthode de Runge-Kutta d'ordre 4
def runge_kutta_4(f, t0, tf, X0, dt, alpha, omega_0):
    t_values = np.arange(t0, tf, dt)
    X_values = np.zeros((len(t_values), len(X0)))
    X_values[0] = X0
    
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        X = X_values[i - 1]
        
        k1 = f(t, X, alpha, omega_0)
        k2 = f(t + dt/2, X + dt*k1/2, alpha, omega_0)
        k3 = f(t + dt/2, X + dt*k2/2, alpha, omega_0)
        k4 = f(t + dt, X + dt*k3, alpha, omega_0)
        
        X_values[i] = X + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t_values, X_values

# Calcul de la solution
t_values, X_values = runge_kutta_4(system, 0, t_final, [x0, v0], dt, alpha, omega_0)

# Calcul des énergies
potential_energy = 0.5 * k * X_values[:, 0]**2
kinetic_energy = 0.5 * mass * X_values[:, 1]**2
total_energy = potential_energy + kinetic_energy

# Création des graphiques avec Plotly
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=('Position vs Temps', 'Vitesse vs Temps',
                                  'Espace des phases', 'Énergies vs Temps'),
                    specs=[[{}, {}],
                          [{}, {}]])

# Position vs Temps
fig.add_trace(
    go.Scatter(x=t_values, y=X_values[:, 0], name="Position",
               line=dict(color='blue')),
    row=1, col=1
)

# Vitesse vs Temps
fig.add_trace(
    go.Scatter(x=t_values, y=X_values[:, 1], name="Vitesse",
               line=dict(color='red')),
    row=1, col=2
)

# Espace des phases
fig.add_trace(
    go.Scatter(x=X_values[:, 0], y=X_values[:, 1], name="Phase",
               line=dict(color='green')),
    row=2, col=1
)

# Énergies
fig.add_trace(
    go.Scatter(x=t_values, y=potential_energy, name="Énergie potentielle",
               line=dict(color='purple')),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=t_values, y=kinetic_energy, name="Énergie cinétique",
               line=dict(color='orange')),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=t_values, y=total_energy, name="Énergie totale",
               line=dict(color='black')),
    row=2, col=2
)

# Mise à jour du layout
fig.update_layout(
    height=800,
    showlegend=True,
    title_text="Analyse du système masse-ressort",
    hovermode='x unified'
)

# Mise à jour des axes
fig.update_xaxes(title_text="Temps (s)", row=1, col=1)
fig.update_xaxes(title_text="Temps (s)", row=1, col=2)
fig.update_xaxes(title_text="Position (m)", row=2, col=1)
fig.update_xaxes(title_text="Temps (s)", row=2, col=2)

fig.update_yaxes(title_text="Position (m)", row=1, col=1)
fig.update_yaxes(title_text="Vitesse (m/s)", row=1, col=2)
fig.update_yaxes(title_text="Vitesse (m/s)", row=2, col=1)
fig.update_yaxes(title_text="Énergie (J)", row=2, col=2)

# Affichage du graphique dans Streamlit
st.plotly_chart(fig, use_container_width=True)

# Affichage des informations supplémentaires
col1, col2 = st.columns(2)

with col1:
    st.subheader("Caractéristiques du système")
    st.write(f"- Période naturelle: {(2*np.pi/omega_0):.2f} s")
    st.write(f"- Fréquence naturelle: {(omega_0/(2*np.pi)):.2f} Hz")
    
with col2:
    st.subheader("Classification du mouvement")
    if damping_ratio > 1:
        st.write("Régime suramorti")
    elif damping_ratio < 1:
        st.write("Régime sous-amorti")
    else:
        st.write("Régime critique")

# Export des données
if st.button("Télécharger les données"):
    df = pd.DataFrame({
        'Temps': t_values,
        'Position': X_values[:, 0],
        'Vitesse': X_values[:, 1],
        'Energie_potentielle': potential_energy,
        'Energie_cinetique': kinetic_energy,
        'Energie_totale': total_energy
    })
    st.download_button(
        label="Télécharger CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='donnees_simulation.csv',
        mime='text/csv'
    )
