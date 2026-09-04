# Die Wrap-Aufgabe: Umschließen eines Zylinders mit einem tendon-getriebenen Tentakel

Dieses Dokument beschreibt die `RlExplor-Spirob-Wrap`-Aufgabe der Simulationsumgebung
im Detail: den physikalischen Aufbau, die Beobachtungs- und Aktionsräume, die
Domain-Randomisierung, und vor allem die Reward-Funktion mathematisch und mit
Begründung ihrer einzelnen Terme. Ein letzter Abschnitt hält den Entwicklungsverlauf
fest — welche Probleme aufgetreten sind, wie sie diagnostiziert und behoben wurden,
und welche Fragen noch offen sind. Alle Angaben beziehen sich auf den Code-Stand in
`rl/src/spirob_rl/tasks/spirob/` zum Zeitpunkt des Schreibens.

## Inhalt

1. [Simulationsumgebung](#1-simulationsumgebung)
2. [Die Wrap-Aufgabe im Detail](#2-die-wrap-aufgabe-im-detail)
3. [Die Reward-Funktion](#3-die-reward-funktion)
4. [Trainingsverfahren](#4-trainingsverfahren)
5. [Entwicklungsverlauf: Probleme, Diagnosen, Erkenntnisse](#5-entwicklungsverlauf-probleme-diagnosen-erkenntnisse)
6. [Offene Fragen](#6-offene-fragen)

---

## 1. Simulationsumgebung

### 1.1 Framework und Architektur

Die Simulation läuft auf [mjlab](https://github.com/mujocolab/mjlab), einem
GPU-parallelisierten RL-Framework auf Basis von MuJoCo-Warp (MJX-artige
Batch-Simulation vieler physikalisch unabhängiger Umgebungen auf der GPU). Ein
einzelner Trainingslauf simuliert typischerweise 1000–4000 Instanzen des Spirob
parallel, jede mit eigenem Zustand, eigener Domain-Randomisierung und eigenem
Zielobjekt, aber ohne physikalische Interaktion zwischen den Instanzen.

Die Aufgabe ist Teil einer Task-Familie (`rl/src/spirob_rl/tasks/spirob/`), die vier
Zielformulierungen auf demselben Tentakel-Modell teilt: *Reach* (statischer
Zielpunkt für die Spitze), *Shape* (Spitze und ein mittleres Segment gleichzeitig),
*Trajectory* (bewegtes Ziel entlang einer Bahn) und *Wrap* (Gegenstand dieses
Dokuments). Gemeinsamer Code — das Tentakel-Modell, die Sehnen-Aktuierung, die
Domain-Randomisierung samt Curriculum, die Sensor-Ablationsstufen und die
Simulationsparameter — liegt in `base_env_cfg.py` und `mdp/`; jede Variante steuert
nur ihre eigenen Kommando- und Reward-Terme bei (`wrap_env_cfg.py` für Wrap).

### 1.2 Physikalisches Modell des Spirob

Der Spirob ist ein planarer, kettenförmiger Tentakel aus 14 quaderförmigen
Segmenten (`seg_13` an der Basis bis `seg_0` an der Spitze), verbunden durch 13
Drehgelenke (`j_12` … `j_0`), deren Achsen alle parallel zur y-Achse liegen. Die
gesamte Bewegung findet daher in der x-z-Ebene statt; y bleibt für jeden Punkt der
Kette stets 0. Diese Einschränkung wird an mehreren Stellen des Codes explizit
ausgenutzt (siehe `mdp/kinematics.py`), u. a. für eine analytische
Vorwärtskinematik ohne MuJoCo-Aufruf.

Jedes Gelenk ist auf $\pm 0.513643\,\mathrm{rad}$ ($\approx \pm 29.4^\circ$)
begrenzt; bei voller gleichsinniger Auslenkung aller 13 Gelenke kann die Kette sich
also um kumulativ bis zu $\approx 6.68\,\mathrm{rad}$ (mehr als eine volle
Umdrehung) einrollen. Die Gelenkdämpfung ist stark ungleich über die Kette verteilt
— sie steigt von $0.0001$ am Basisgelenk `j_12` (praktisch ungedämpft) auf
$\approx 121.3$ am Spitzengelenk `j_0`. Dieser Gradient ist Teil der ursprünglichen
Modellkalibrierung (nicht in dieser Arbeit verändert) und hat direkte Konsequenzen
für die numerische Stabilität der Simulation (siehe Abschnitt 5.1).

Für jedes Segment definiert die Modell-XML eine Site `site_imu_<n>` (`n`= 0…13) am
geometrischen Mittelpunkt des Segments — ursprünglich als Platzhalter für
IMU-Sensorik gedacht, in dieser Arbeit aber zusätzlich als Trackingpunkt für
Positions- und Kontaktauswertung genutzt (`IMU_CFG` in `mdp/constants.py`). Die
Spitze der Kette trägt eine eigene Site `site_tcp`.

### 1.3 Aktuierung

Der Spirob wird ausschließlich über zwei antagonistische Sehnen (`tendon_0`,
`tendon_1`) angetrieben, die durch alle 14 Segmente hindurchlaufen und an der
Spitze verankert sind. Jede Sehne kann nur ziehen, nicht drücken: Der zugehörige
MuJoCo-Aktuator hat `ctrlrange = (-150, 0)` (Einheit Newton, negatives Vorzeichen
= Zugkraft). Die Policy gibt pro Sehne einen unbeschränkten reellen Wert aus, der
über eine affine Abbildung

$$
u_i = 75 \cdot a_i - 75, \qquad a_i \in \mathbb{R}
$$

auf die Kraft $u_i \in [-150, 0]\,\mathrm{N}$ abgebildet wird (`TENDON_FORCE_SCALE
= 75`, `TENDON_FORCE_OFFSET = -75`), wobei zusätzlich hart auf $[-150, 0]$
geklemmt wird. Für $a_i \in [-1, 1]$ deckt das genau den physikalischen
Stellbereich ab; außerhalb dieses Intervalls sättigt die Aktuierung, ohne dass
sich die Simulation weiter ändert (relevant für Abschnitt 5.2).

Mit **13 Gelenken, aber nur 2 unabhängigen Stellgrößen** ist der Spirob strukturell
unteraktuiert. Die Menge der bei gegebenem Sehnenzug *statisch haltbaren* Posen
bildet daher höchstens eine 2-Mannigfaltigkeit im 13-dimensionalen Gelenkraum —
das Bild des 2D-Stellgrößenquadrats unter der quasistatischen Abbildung. Diese
Eigenschaft wird in der Shape-Aufgabe der Task-Familie explizit vermessen
(`HoldablePoseTable`) und ist auch für die Wrap-Aufgabe relevant, weil sie erklärt,
warum das System multistabil ist: Dasselbe Sehnenkommando kann je nach
Ausgangskonfiguration in unterschiedlichen Endlagen einrasten (empirisch
verifiziert: bei einem 12×12-Gitter über den Stellgrößenraum landeten
Konfigurationen aus unterschiedlichen Startzuständen in $85\,\%$ der Fälle mehr
als $5\,\mathrm{cm}$ auseinander).

### 1.4 Simulationsparameter

| Parameter | Wert |
|---|---|
| Zeitschritt (Physik) | $0.004\,\mathrm{s}$ |
| Decimation (Policy-Schritte pro Physik-Schritt) | 5 |
| effektive Policy-Frequenz | $50\,\mathrm{Hz}$ |
| Episodenlänge | $30\,\mathrm{s}$ ($1500$ Policy-Schritte) |
| Integrator / Solver | wie in `spirob.xml` (`impratio=18.78`, `cone=elliptic`, `iterations=20`) |

### 1.5 Domain-Randomisierung und Curriculum

Sieben physikalische Parameter (Gelenksteifigkeit, -dämpfung, -reibung;
Sehnensteifigkeit, -dämpfung, -reibungsverlust; siehe `DR_TARGETS` in
`mdp/constants.py`) werden bei jedem Reset multiplikativ randomisiert
(`operation="scale"`): Der tatsächliche Wert ist der XML-Nominalwert multipliziert
mit einem pro Umgebung gezogenen Faktor. Weil skaliert statt absolut überschrieben
wird, bleibt das ursprüngliche Gefälle der Werte über die Kette erhalten (z. B. der
oben beschriebene Dämpfungsgradient), nur die Bandbreite variiert.

Die Bandbreite selbst wird nicht von Beginn an ausgeschöpft, sondern über ein
Curriculum linear aufgezogen (`dr_range_curriculum` in `mdp/curriculums.py`):

$$
\alpha(t) = \operatorname{clip}\!\left(\frac{t - t_0}{t_1 - t_0},\ 0,\ 1\right),
\qquad
\text{range}(t) = (1-\alpha(t)) \cdot (1, 1) + \alpha(t) \cdot \text{range}_{\text{final}}
$$

mit $t$ = Anzahl der bereits ausgeführten Policy-Schritte, $t_0 = 0$,
$t_1 = 5000$. Zu Trainingsbeginn ($\alpha = 0$) entspricht jeder Faktor exakt $1.0$
— die Simulation läuft also zunächst bei den nominalen XML-Werten ohne jede
Streuung — und erreicht nach 5000 Policy-Schritten die volle konfigurierte
Bandbreite (z. B. Gelenkdämpfung zwischen $10\,\%$ und $100\,\%$ des Nominalwerts).
Die Motivation ist die aus dem Sim-to-Real-Bereich bekannte Domain-Randomization-
Curriculum-Praxis: Die Policy soll zunächst eine grundsätzlich funktionierende
Strategie unter „einfachen“, deterministischen Bedingungen finden, bevor sie mit
der vollen Parameterstreuung robust gemacht wird.

### 1.6 Beobachtungsraum: Sensor-Ablationsstufen

Die Task-Familie ist so gebaut, dass sich systematisch untersuchen lässt, wie viel
Sensorinformation eine Policy zum Lösen der Aufgabe tatsächlich braucht — relevant,
weil die Zielhardware (ein reales Tentakel-Rig) nicht jede in der Simulation
verfügbare Größe messen kann. Fünf Stufen (`SensorLevel` in `base_env_cfg.py`),
streng geschachtelt:

| Stufe | Zusätzliche Beobachtungen | Auf der Hardware messbar? |
|---|---|---|
| `force` | (nichts; nur Ziel + letzte Aktion) | — |
| `tendon` | Sehnenlänge, Sehnengeschwindigkeit | ja (Spulenencoder) |
| `imu` | Segment-Neigungswinkel (cos/sin) | ja (IMU pro Segment) |
| `joints` | alle 13 Gelenkwinkel und -geschwindigkeiten | ja (Beschleunigungssensor-Board) |
| `oracle` | zusätzlich die exakte TCP-Position | nein (privilegiert) |

Der Critic bekommt in jeder Konfiguration den vollen privilegierten Zustand
(Gelenkwinkel, -geschwindigkeiten, TCP-Position, Ziel, letzte Aktion) — nur der
Actor wird auf die jeweilige Stufe beschränkt. Für die in diesem Dokument
diskutierten Trainingsläufe wurde überwiegend die Stufe `joints` verwendet.

Die Aktor-Beobachtung enthält zusätzlich eine gleitende Historie der letzten $H=5$
bis $10$ Zeitschritte (`history_length`), weil eine einzelne Momentaufnahme (z. B.
nur die aktuelle Sehnenlänge) die Systemdynamik nicht vollständig bestimmt — der
Tentakel ist bei gegebener Sehnenlänge nicht eindeutig geformt, wohl aber bei
gegebenem kurzen Zeitfenster.

---

## 2. Die Wrap-Aufgabe im Detail

### 2.1 Zielsetzung

Bei jedem Reset erscheint ein Zylinder an einer zufälligen Position mit
zufälligem Radius in einem definierten Bereich. Die Policy soll den Tentakel so
steuern, dass er sich um den Zylinder legt — nicht nur mit der Spitze berührt,
sondern mit möglichst vielen Segmenten entlang der Kette Kontakt herstellt, und
zwar so, dass sich diese Kontakte (und die dabei übertragene Kraft) gleichmäßig um
den Zylinder verteilen statt sich an einer Stelle zu konzentrieren.

### 2.2 Das Objekt: ein kollisionsfähiger, kinematisch fixierter Zylinder

Das Zielobjekt ist eine eigene, vom Spirob unabhängige mjlab-Entity
(`mdp/object_spec.py`): ein einzelner Körper mit einem Zylinder-Geom, ohne
Gelenk. mjlab verpackt gelenklose Entities automatisch in einen *Mocap*-Body
(`mjlab.utils.spec.auto_wrap_fixed_base_mocap`) — dessen Pose wird nicht über die
Gelenkkette, sondern direkt über `Entity.write_mocap_pose_to_sim(...)` gesetzt,
pro Umgebung individuell und ohne dass eine Simulationsdynamik (Trägheit,
Impuls) daran hängt. Der Zylinder bleibt dadurch physikalisch **fixiert** — wie
ein einbetonierter Pfahl — reagiert also nicht auf Kontaktkräfte, generiert aber
selbst reale Normal- und Reibungskräfte gegen alles, was ihn berührt.

Das Zylinder-Geom wird über MuJoCos `fromto`-Spezifikation definiert, sodass seine
Längsachse unabhängig von der Körperorientierung entlang der lokalen y-Achse
verläuft — senkrecht zur Bewegungsebene des Tentakels, wie eine horizontale Stange,
um die sich der Tentakel in der x-z-Ebene wickelt. Kollisionseigenschaften
(`contype`, `conaffinity`, Reibung, `solref`/`solimp`) werden bewusst **nicht**
gesetzt, sondern bleiben auf MuJoCo-Standardwerten — konsistent damit, dass auch
die Segment-Geome der Spirob-XML keine dieser Eigenschaften individuell
überschreiben.

Radius und feste Halblänge:

$$
r_{\text{obj}} \sim \mathcal{U}(0.015,\ 0.10)\,\mathrm{m}, \qquad
\ell_{\text{obj}} = 0.08\,\mathrm{m} \ (\text{fix})
$$

Der Radiusbereich wurde im Projektverlauf von ursprünglich $(0.015, 0.05)$ auf
$(0.015, 0.10)$ verdoppelt (siehe Abschnitt 5.5); die Halblänge ist so gewählt,
dass sie die y-Halbausdehnung der Segmente ($\approx 0.046\,\mathrm{m}$ an der
Basis) überragt, damit kein Segment seitlich am Zylinder vorbeirutschen kann.

### 2.3 Kommando: Position und Radius (`WrapCommand`)

`WrapCommand` (`mdp/commands.py`) ist die einzige Kommando-Klasse der Task-Familie,
deren „Ziel“ kein Punkt auf dem Spirob selbst ist, sondern ein reales Objekt in
der Szene. Sie hält zwei Zustandsgrößen als *single source of truth*:
`target_pos_w` (Zylindermittelpunkt, Weltkoordinaten) und `target_radius`.

**Positionssampling.** Der Mittelpunkt wird in Polarkoordinaten relativ zur Basis
gezogen — Winkel $\varphi$ von der +z-Achse, Abstand $\rho$ von der Basis:

$$
\varphi \sim \mathcal{U}(-1.0, -0.6)\,\mathrm{rad}, \qquad
\rho \sim \mathcal{U}(0.24, 0.32)\,\mathrm{m}
$$

$$
\mathbf{c} = \big(\rho \sin\varphi,\ 0,\ \rho \cos\varphi\big)
$$

Diese Koordinatenkonvention ($x = \rho\sin\varphi$, $z = \rho\cos\varphi$) ist
identisch zur Zielpunkt-Parametrisierung der Reach-Aufgabe und zieht sich durch
den gesamten Code (`polar_to_xz` in `mdp/commands.py`).

**Radiussampling.** Unabhängig davon: $r_{\text{obj}} \sim \mathcal{U}(0.015,
0.10)\,\mathrm{m}$.

**Anwendung auf die Simulation.** Die gezogene Position wird per
`write_mocap_pose_to_sim` gesetzt (mit Additions des jeweiligen
`env_origins`-Offsets, der für die Gitteranordnung paralleler Umgebungen bei der
Visualisierung sorgt, siehe `mdp/events.py::grid_layout` — für die Physik/den
Reward irrelevant, da er sich in jeder Differenz heraushebt). Der Radius wird
direkt in das kompilierte Modellfeld `geom_size` pro Umgebung geschrieben, mit
derselben Technik der Pro-Welt-Feld-Expansion, die auch `grid_layout` für die
Basisposition des Spirob nutzt.

### 2.4 Reset-Design: kollisionsfreier Start durch Vorzeichenkonstruktion

Der Reset des Tentakels ist bewusst **nicht** symmetrisch zufällig, sondern
einseitig gebiaст, und zwar **von der Objektseite weg**:

$$
q_i^{(0)} \sim \mathcal{U}(0.0,\ 0.5)\,\mathrm{rad}, \qquad i = 1, \dots, 13
\quad \text{(unabhängig pro Gelenk)}
$$

während der Zylinder bei $\varphi \in (-1.0, -0.6)$, also auf der $-x$-Seite,
erscheint. Das ist kein Zufall, sondern eine geometrisch beweisbare
Sicherheitseigenschaft: In der Vorwärtskinematik gilt für die $x$-Koordinate jedes
Kettenpunkts

$$
x = \sum_{k} \ell_k \sin(\theta_k), \qquad \theta_k = \sum_{i \le k} q_i
$$

d. h. $x$ ist eine Summe von Termen $\sin(\theta_k)$ mit *kumulativ* aufsummierten
Winkeln $\theta_k$. Sind **alle** $q_i \ge 0$, so ist auch jedes $\theta_k \ge 0$,
und für $\theta_k \in [0, \pi]$ (was der begrenzte Gelenkraum bei Reset-typischen
Werten garantiert) folgt $\sin(\theta_k) \ge 0$ und damit $x \ge 0$ für die gesamte
Kette. Der Zylinder liegt jedoch vollständig bei $x < 0$ — die Reset-Pose kann ihn
also *geometrisch* nicht erreichen, nicht nur „mit hoher Wahrscheinlichkeit“.

Diese Eigenschaft galt zunächst nur für den Mittelpunkt des Zylinders; nach der
Verdopplung des Radius (Abschnitt 5.5) musste zusätzlich sichergestellt werden,
dass auch die *Objektoberfläche* — nicht nur ihr Zentrum — weder in die
$x\ge 0$-Reset-Region noch unter die Bodenebene ($z=0$) hineinragt. Beides wurde
empirisch über $1024$ parallele Umgebungen verifiziert (minimaler
Oberflächenabstand zur Reset-Pose $> 7\,\mathrm{cm}$, minimaler Bodenabstand
$> 4\,\mathrm{cm}$).

### 2.5 Synchronisation von Objekt- und Gelenk-Reset

Damit die geometrische Garantie aus 2.4 tatsächlich bei jedem Reset gilt, müssen
Gelenkstellung und Objektposition **gleichzeitig** neu gezogen werden — sonst
könnte ein frisch zurückgesetzter Tentakel auf ein „altes“ Objekt treffen. Ein
Blick in den mjlab-Quellcode (`ManagerBasedRlEnv._reset_idx`) bestätigt, dass dies
strukturell garantiert ist: Jeder Reset ruft zunächst das `reset_joints`-Event auf
und danach unbedingt `command_manager.reset()` für jedes aktive Kommando.
`CommandTerm.reset()` erzwingt dabei ein sofortiges `_resample_command()`,
unabhängig vom eigenen `resampling_time_range`-Zähler. Objekt und Gelenke sind
also bei jedem Environment-Reset synchron.

Diese Synchronisation gilt jedoch **nur** für den Reset, nicht für ein
zwischenzeitliches Neu-Sampling, das `resampling_time_range` normalerweise
zusätzlich erlaubt (bei den anderen Aufgaben der Task-Familie üblich, um z. B.
mehrere Zielpunkte pro Episode zu üben). Für Wrap wurde das bewusst deaktiviert
(`resampling_time_range=(1.0e6, 1.0e6)`, siehe Abschnitt 5.4) — bei den anderen
Aufgaben ist das Ziel ein abstrakter Punkt ohne physische Präsenz, ein
Neu-Sampling mitten in der Episode also harmlos. Der Wrap-Zylinder ist dagegen ein
reales, kollisionsfähiges Objekt: Ein Sprung an eine neue Position, während der
Tentakel noch an der alten Stelle steht, hätte ihn direkt in den Tentakel hinein
teleportiert.

---

## 3. Die Reward-Funktion

### 3.1 Designphilosophie

Der Reward ist bewusst additiv aus mehreren Termen mit unterschiedlicher
„Härte“ zusammengesetzt — eine in der RL-Praxis übliche Strategie, um das
Lernproblem in aufeinander aufbauende Teilziele zu zerlegen, statt ein einziges,
schwer zu optimierendes Endziel direkt zu belohnen:

1. **`wrap_proximity`** (grob) — bringe irgendetwas in die Nähe des Zylinders.
2. **`wrap_proximity_fine`** (fein) — schmiege die Oberfläche eng an.
3. **`wrap_coverage`** — verteile die *Positionen* der nahen Segmente um den
   Zylinder statt sie auf einer Seite zu konzentrieren.
4. **`wrap_force_distribution`** — verteile die tatsächlich gemessene
   Kontakt*kraft* gleichmäßig, nicht nur die Positionen.

Die ersten drei Terme sind rein geometrisch (basierend auf den Site-Positionen der
Segmente relativ zum kommandierten Zylinderzentrum) und erfordern keine
Kollisionsphysik. Der vierte Term liest reale, physikalisch simulierte
Kontaktkräfte über einen `ContactSensor` und ist damit der einzige Term, der ohne
tatsächliche Kollision (Abschnitt 2.2) gar nicht sinnvoll definiert wäre.

Zusätzlich existieren drei Regularisierungsterme (`action_rate`,
`action_magnitude`, `joint_vel`), die im aktuellen Code-Stand **auskommentiert**
sind (siehe `wrap_env_cfg.py`) — ihre Funktion und der Grund für ihre Existenz
werden in Abschnitt 3.5 dennoch dokumentiert, da sie für die Trainingsstabilität
nachweislich relevant waren (Abschnitt 5.2) und bei Bedarf reaktiviert werden
können.

### 3.2 `wrap_proximity` und `wrap_proximity_fine`

Für jede der $P=14$ Segment-Sites $\mathbf{s}_p$ wird der Abstand zur
Zylinderoberfläche berechnet:

$$
d_p = \lVert \mathbf{s}_p - \mathbf{c} \rVert_2 - r_{\text{obj}}
$$

($d_p = 0$ heißt exakt auf der Oberfläche, $d_p > 0$ außerhalb, $d_p < 0$
geometrisch „innerhalb“ des Zylinderradius — was bei echter Kollision nur als
elastische Deformation/Eindringtiefe im Kontaktlöser vorkommt, nicht als
tatsächliche Durchdringung). Der Reward ist ein über alle Segmente gemittelter
Gauß-Kernel:

$$
r_{\text{prox}} = \frac{1}{P} \sum_{p=1}^{P} \exp\!\left(-\frac{d_p^2}{\sigma^2}\right)
$$

mit zwei Instanzen dieses Terms bei unterschiedlicher Standardabweichung
$\sigma$: $\sigma = 0.08\,\mathrm{m}$ (`wrap_proximity`, Gewicht $1.0$) für ein
weiträumig wirksames Gradientensignal in der frühen Trainingsphase, und
$\sigma = 0.03\,\mathrm{m}$ (`wrap_proximity_fine`, Gewicht $1.0$) für ein enges
Andocken, sobald die grobe Annäherung gelungen ist. Dieses Muster
(grob + fein, gleiche Kernfunktion mit zwei $\sigma$) wird in der gesamten
Task-Familie wiederverwendet (`position_tracking` in `mdp/rewards.py` für Reach,
Shape, Trajectory).

**Wichtig für die Interpretation:** Der Mittelwert über *alle* 14 Segmente
(nicht nur das nächstgelegene) ist die entscheidende Design-Entscheidung, die
diesen Term von einer reinen Erreich-Aufgabe unterscheidet — ein Segment allein
kann den Reward nicht sättigen; er wächst mit der *Anzahl* der nahen Segmente.
Er sagt jedoch nichts über deren *Verteilung um* den Zylinder aus (dafür ist
`wrap_coverage` zuständig) und — entscheidend, siehe Abschnitt 5.6 — er basiert
auf einem einzelnen Punkt pro Segment (der `site_imu`-Site am geometrischen
Mittelpunkt), nicht auf der tatsächlichen Quader-Kollisionsgeometrie. Ein kleiner
$d_p$ bedeutet daher nicht zwangsläufig echten physischen Kontakt.

### 3.3 `wrap_coverage`: Zirkularstatistik der Winkelverteilung

Dieser Term soll unterscheiden, ob die *nahen* Segmente gleichmäßig um den
Zylinder verteilt sind oder alle auf einer Seite kleben. Dazu wird jedem Segment
zunächst derselbe Näherungs-Kernel wie in 3.2 als Gewicht zugeordnet,

$$
w_p = \exp\!\left(-\frac{d_p^2}{\sigma_c^2}\right), \qquad \sigma_c = 0.05\,\mathrm{m}
$$

und sein Winkel um den Zylindermittelpunkt in der x-z-Ebene bestimmt,

$$
\theta_p = \operatorname{atan2}(x_p - c_x,\ z_p - c_z)
$$

(dieselbe Winkelkonvention wie beim Kommando-Sampling). Aus diesen gewichteten
Winkeln wird der **gewichtete resultierende Vektor** gebildet — eine
Standardgröße der zirkulären Statistik:

$$
\bar{R}_x = \frac{\sum_p w_p \cos\theta_p}{\sum_p w_p}, \qquad
\bar{R}_z = \frac{\sum_p w_p \sin\theta_p}{\sum_p w_p}, \qquad
R = \sqrt{\bar{R}_x^2 + \bar{R}_z^2}
$$

$R \in [0, 1]$ ist die *mittlere Resultantenlänge*: Sind die gewichteten
Winkel gleichmäßig über den vollen Kreis verteilt, heben sich die Einheitsvektoren
gegenseitig auf und $R \to 0$; liegen alle Winkel nahe beieinander (Konzentration
auf einer Seite), verstärken sie sich und $R \to 1$. Der Reward ist ihr
Komplement, zusätzlich skaliert mit dem mittleren Gewicht:

$$
r_{\text{cov}} = (1 - R) \cdot \frac{1}{P}\sum_{p} w_p
$$

Die Skalierung mit $\frac{1}{P}\sum_p w_p$ ist notwendig, damit der Term nicht
für „weit verstreute, aber weit entfernte“ Segmente hoch ausfällt — ohne sie wäre
$1-R$ auch dann groß, wenn kein Segment in der Nähe des Zylinders ist, die
Winkel $\theta_p$ aber zufällig gleichverteilt sind (was bei großer Entfernung
generisch der Fall wäre). Gewichtsnormierung: `weight_sum.clamp_min(1e-6)`
verhindert Division durch Null, wenn kein Segment überhaupt Gewicht $>0$ trägt.

### 3.4 `wrap_force_distribution`: Shannon-Entropie der Kontaktkraftverteilung

Dieser Term wurde nachträglich hinzugefügt, nachdem sich zeigte (Abschnitt 5.6),
dass `wrap_proximity`/`wrap_coverage` — weil sie auf Punktabständen statt auf
echter Kollisionsgeometrie beruhen — eine Policy nicht zuverlässig zu **echtem,
verteiltem physischem Kontakt** anleiten: Eine trainierte Policy konnte die
geometrischen Terme weitgehend erfüllen ($d_p$ klein für viele Segmente), während
der tatsächliche `ContactSensor` zeigte, dass real fast die gesamte Kontaktkraft
über ein einzelnes Segment übertragen wurde.

**Sensorik.** Ein `ContactSensorCfg` (`wrap_env_cfg.py`) misst pro Segment-Geom
(`primary`, Muster `g_0` … `g_13`) die Nettokraft gegen das Objekt-Geom
(`secondary`), mit `reduce="netforce"` — alle an einem Segment gleichzeitig
auftretenden Kontaktpunkte werden zu einem resultierenden Kraftvektor
aufsummiert, statt nur den stärksten Einzelkontakt zu behalten (relevant, weil
ein Quader-gegen-Zylinder-Kontakt an Kanten/Ecken potenziell mehrere
Berührpunkte gleichzeitig erzeugt). Ergebnis pro Umgebung: ein Kraftvektor
$\mathbf{f}_p \in \mathbb{R}^3$ für jedes der $P=14$ Segmente.

**Reward.** Aus den Kraftbeträgen $m_p = \lVert \mathbf{f}_p \rVert_2$ wird eine
normierte Verteilung gebildet,

$$
p_p = \frac{m_p}{\sum_{k=1}^{P} m_k}, \qquad p_p \ge 0,\ \sum_p p_p = 1
$$

und deren **Shannon-Entropie** berechnet:

$$
H = -\sum_{p=1}^{P} p_p \ln p_p
$$

(mit der Konvention $0 \ln 0 := 0$, mathematisch der Grenzwert
$\lim_{x\to 0} x\ln x = 0$, sodass Segmente ohne Kontakt sauber aus der Summe
herausfallen). Normiert auf $[0, 1]$:

$$
r_{\text{force}} = \frac{H}{\ln P}, \qquad P = 14
$$

wobei $r_{\text{force}} := 0$ gesetzt wird, wenn die Gesamtkraft
$\sum_p m_p$ unter einer Schwelle ($10^{-3}\,\mathrm{N}$) liegt (kein
Kontakt → Reward exakt 0, nicht undefiniert).

**Warum Entropie und nicht z. B. Varianz oder Gini-Koeffizient?** Die
Shannon-Entropie einer Verteilung über einen festen Träger der Größe $P$ ist
maximal genau dann, wenn die Verteilung uniform über den *gesamten* Träger ist
($p_p = 1/P\ \forall p$, $H = \ln P$), und minimal bei vollständiger
Konzentration auf ein Element ($H = 0$, unabhängig von $P$). Für eine
Konzentration auf $k < P$ Elemente mit dort gleicher Kraft gilt
$H = \ln k$. Damit steigt der normierte Reward $r_{\text{force}} = \ln k / \ln P$
**sowohl** mit zunehmender Gleichverteilung *unter den aktuell aktiven Kontakten*
**als auch** mit der *Anzahl* der aktiven Kontakte — eine einzige Formel erfasst
beide vom Nutzer gewünschten Eigenschaften („möglichst viele Segmente Kontakt“
und „Kraft möglichst gleichverteilt“), ohne zwei separate, gegeneinander zu
gewichtende Terme zu benötigen. Eine reine Streuungsmaßzahl (z. B. der
Variationskoeffizient $\sigma_f/\mu_f$ der Kräfte) hätte diese Doppelrolle nicht
in derselben Weise: Sie bewertet nur die Gleichmäßigkeit unter den *ohnehin*
belasteten Segmenten, ignoriert aber, wie viele das überhaupt sind.

**Eigenschaften und Grenzen dieser Formulierung** (für die wissenschaftliche
Einordnung wichtig):

* *Skaleninvarianz*: $r_{\text{force}}$ hängt nur von den *Proportionen* $p_p$
  ab, nicht von der absoluten Kraft. Ein sehr fest und ein sehr locker, aber
  gleichmäßig verteilter Griff erhalten denselben Reward. Das ist beabsichtigt
  (der Nutzer fragte explizit nach der *Verteilung*, nicht nach der
  Griffkraft), aber eine bewusste Einschränkung: Der Term allein motiviert
  keine Mindestgriffkraft.
* *Keine explizite Topologie*: Die Formel bewertet die Winkelverteilung der
  Kraftbeträge nicht direkt (anders als `wrap_coverage`) — sie könnte
  theoretisch von einer Verteilung über *irgendeine* Teilmenge von Segmenten
  ebenso hoch bewertet werden wie von einer tatsächlich umschließenden
  Verteilung. In der Praxis ist dieses Risiko durch das Zusammenspiel mit
  `wrap_coverage` (das die räumliche Verteilung bereits erzwingt) begrenzt,
  wurde aber nicht separat isoliert getestet.
* *Verifikation der Implementierung*: Die Entropie-Berechnung wurde gegen
  Hand-konstruierte Kraftverteilungen geprüft (siehe Abschnitt 5.6): eine
  Konzentration auf 1 Segment ergibt $r_{\text{force}} = 0$, Gleichverteilung
  über 2 Segmente $r_{\text{force}} = \ln 2 / \ln 14 = 0.2626$, Gleichverteilung
  über alle 14 Segmente $r_{\text{force}} = 1.0$ — exakt wie erwartet.

### 3.5 Regularisierungsterme (aktuell deaktiviert)

Drei zusätzliche Terme sind im Code vorhanden, aber im aktuellen Stand von
`wrap_env_cfg.py` auskommentiert:

* **`action_magnitude`** ($\propto \lVert \mathbf{a} \rVert_2^2$, rohe,
  unskalierte Policy-Aktion): Ohne diesen Term kann der Mittelwert der
  Policy-Verteilung beliebig weit über den Sättigungsbereich
  $a \in [-1, 1]$ hinauswandern, ohne dass sich die Simulation dadurch noch
  ändert (Abschnitt 5.2) — es gibt dann kein physikalisches Rückstellsignal.
* **`action_rate`** ($\propto \lVert \mathbf{a}_t - \mathbf{a}_{t-1} \rVert_2^2$,
  mjlab-Standardterm): bestraft sprunghafte Aktionsänderungen.
* **`joint_vel`** ($\propto \lVert \dot{\mathbf{q}} \rVert_2^2$): dämpft
  überschüssige Gelenkgeschwindigkeit.

Diese Terme wurden ursprünglich eingeführt, um eine spezifische
Trainingsdivergenz zu verhindern (Abschnitt 5.2) und sind seither Teil des
gemeinsamen Reward-Vokabulars der Task-Familie. Dass sie für die Wrap-Aufgabe
aktuell ausgeschaltet sind, ist eine bewusste, vom Nutzer gesetzte
Experimentierkonfiguration — die Stabilisierung gegen Aktionssättigung erfolgt in
der aktuellen Konfiguration stattdessen auf einer anderen Ebene (`clip_actions`,
Abschnitt 5.2).

### 3.6 Gesamt-Reward

$$
r = 1.0 \cdot r_{\text{prox}}(\sigma{=}0.08)
  + 1.0 \cdot r_{\text{prox}}(\sigma{=}0.03)
  + 1.5 \cdot r_{\text{cov}}
  + 1.0 \cdot r_{\text{force}}
$$

Die Gewichtsverhältnisse spiegeln eine bewusste Rangfolge wider: `wrap_coverage`
ist am höchsten gewichtet, weil es die räumliche Verteilung als Voraussetzung für
jede Form von „Umschließen“ erzwingt; `wrap_force_distribution` ist niedriger
gewichtet, weil es nur dann ein sinnvolles Signal liefert, wenn überhaupt Kontakt
besteht — die geometrischen Terme müssen diesen Kontakt erst herstellen, der
Kraftterm verfeinert ihn nur.

---

## 4. Trainingsverfahren

Trainiert wird mit PPO (Implementierung: `rsl_rl`, über mjlab). Aktor und Critic
sind separate MLPs mit Architektur $128\text{-}128\text{-}128\text{-}128\text{-}64$
und ELU-Aktivierung; die Aktionsverteilung ist eine Gauß-Verteilung mit
skalarer, gelernter Standardabweichung (Startwert $2.0$). Zentrale
Hyperparameter:

| Parameter | Wert |
|---|---|
| `value_loss_coef` | 1.0 |
| `clip_param` (PPO) | 0.3 |
| `entropy_coef` | 0.05 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `learning_rate` | $10^{-3}$, adaptives Schedule nach KL |
| `desired_kl` | 0.01 |
| `gamma` / `lam` (GAE) | 0.99 / 0.95 |
| `max_grad_norm` | 1.0 |
| `num_steps_per_env` | 64 |
| `max_iterations` | 200 |
| `clip_actions` | 5.0 (siehe Abschnitt 5.2) |

Zusätzlich wurden die MuJoCo-Warp-Kontaktpuffer explizit vergrößert
(`nconmax=300`, `njmax=400`, Standard wäre ein interner Heuristikwert) — Grund
und Herleitung in Abschnitt 5.1.

---

## 5. Entwicklungsverlauf: Probleme, Diagnosen, Erkenntnisse

Dieser Abschnitt hält den tatsächlichen iterativen Entwicklungsprozess fest —
nicht nur das Endergebnis, sondern auch die Fehlschläge und was aus ihnen
gelernt wurde, da dies für die Methodik-Diskussion einer wissenschaftlichen
Arbeit relevant sein dürfte.

### 5.1 Kollisions-Constraint-Überlauf (`nconmax`/`njmax`)

Nach Aktivierung echter Kollision (Abschnitt 2.2) divergierte ein
Trainingslauf reproduzierbar um Iteration 61: Der Value-Loss stieg innerhalb
weniger Iterationen von einem gesunden Bereich über $10^4$, $10^8$ bis
$\infty$. Ein reiner Physik-Rollout ohne RL-Update (2000 Schritte,
Zufallsaktionen) zeigte in den Logs wiederholt `nefc overflow — please
increase njmax`: MuJoCo-Warp reserviert pro paralleler Welt einen festen
Speicherblock für Kontakt-Constraints (`nconmax`) und generalisierte
Zwangsbedingungen (`njmax`); ohne explizite Angabe wird ein interner
Heuristikwert verwendet, der für die Kombination aus Segment-Selbstkontakt
(bereits im Basismodell vorhanden) und dem zusätzlichen, bei engem Umschließen
potenziell an vielen Segmenten gleichzeitig aktiven Objektkontakt zu klein
war. Werden mehr Constraints aktiv als reserviert, verwirft der Solver
überzählige Einträge kommentarlos — die resultierenden Kontaktkräfte sind
physikalisch inkonsistent, was sich in aufschaukelnden Geschwindigkeiten und
letztlich im explodierenden Value-Loss niederschlägt.

**Fix:** `cfg.sim.nconmax = 300`, `cfg.sim.njmax = 400`, großzügig über dem in
einem 2000-Schritte-Stresstest beobachteten tatsächlichen Bedarf (deutlich
unter 100) angesetzt. Nach dem Fix lief derselbe Trainingslauf über 220
Iterationen ohne jede Instabilität.

### 5.2 Aktionssättigung und `clip_actions`

Auch nach Fix 5.1 trat ein strukturell ähnlicher, aber ursächlich anderer
Absturz erneut auf — diesmal bei voller Domain-Randomisierungsbreite
($\alpha=1.0$, siehe Abschnitt 1.5), reproduzierbar um Iteration 100 eines
länger laufenden Trainings. Reine Physik-Rollouts (sowohl mit Zufallsaktionen
als auch angetrieben durch eine bereits trainierte, objektsuchende Policy)
blieben unter denselben Bedingungen stabil — die Instabilität entstand also
nicht in der Physik selbst, sondern im PPO-Update.

Ursache: `TendonEffortActionCfg` begrenzt zwar die *physikalisch wirksame*
Stellgröße (nach der affinen Abbildung, Abschnitt 1.3) hart auf
$[-150, 0]\,\mathrm{N}$, nicht aber die *rohe* Policy-Ausgabe $a_i$, auf der
die (aktuell deaktivierten, siehe 3.5) Regularisierungsterme
`action_rate`/`action_magnitude` rechnen. Läuft der Mittelwert der
Aktionsverteilung durch einen ungünstigen Gradientenschritt über
$|a_i| > 1$ hinaus, ändert sich die Simulation nicht mehr (Sättigung), es
gibt also kein physikalisches Rückstellsignal — während der quadratische
Strafterm auf einer immer weiter wachsenden rohen Aktion unbegrenzt weiter
zunimmt. Bei voller Domain-Randomisierung (mehr Kontakt, geringere Dämpfung
→ größere, verrauschtere Policy-Gradienten) reichte ein einzelner
ungünstiger Schritt, um diesen Effekt auszulösen.

**Fix:** `clip_actions=5.0` auf Ebene des `RslRlVecEnvWrapper` — ein
harter, vom Environment unabhängig durchgesetzter Clamp auf die rohe Aktion
*vor* jedem Simulationsschritt, statt nur ein statistischer Gegendruck über
Reward-Gewichte. Da diese Einstellung im gemeinsamen `make_ppo_runner_cfg`
gesetzt ist, gilt sie für alle vier Aufgaben der Task-Familie, nicht nur für
Wrap.

### 5.3 Geometrische Sicherheit des Resets

Der ursprüngliche Reset-Bias war (versehentlich) so gepolt, dass der Tentakel
beim Zurücksetzen *zur* Objektseite hin geneigt startete. Nach Korrektur
(Abschnitt 2.4) wurde die Sicherheitseigenschaft „Reset-Pose kann den Zylinder
nicht überlappen“ nicht nur behauptet, sondern über 1024 parallele Umgebungen
empirisch verifiziert (minimaler Oberflächenabstand $> 0$ in jedem Fall).

### 5.4 Kollision während der Episode: Objekt-Neusampling mitten in der Episode

Bei der visuellen Kontrolle (Viser-Viewer) fiel auf, dass der Zylinder
gelegentlich neu positioniert wurde, während der Tentakel an seiner aktuellen
(von der Policy kontrollierten) Position verblieb — mit der Folge, dass der
neue Zylinder direkt in den Tentakel hinein materialisierte oder der Tentakel
unterhalb des Objekts hängen blieb. Ursache: `WrapCommandCfg` erbte den für
die anderen Aufgaben sinnvollen Mechanismus, Kommandos nicht nur bei jedem
Reset, sondern zusätzlich in einem eigenen Intervall (`resampling_time_range`)
neu zu ziehen — für abstrakte Zielpunkte harmlos, für ein reales
kollisionsfähiges Objekt nicht.

**Fix:** `resampling_time_range=(1.0e6, 1.0e6)` — der interne Zeitgeber der
`WrapCommand` läuft praktisch nie ab; die einzige verbleibende
Aktualisierungsquelle ist der garantiert synchrone Reset (Abschnitt 2.5).
Verifiziert: Objektdrift $0.0$ über 500 Schritte innerhalb einer Episode,
korrekte Neupositionierung exakt beim Episodenende.

### 5.5 Radiusverdopplung und Rekalibrierung der Spawn-Region

Die Verdopplung des maximalen Zylinderradius (Abschnitt 2.2) brach beide in
5.3 und 5.4 etablierten Sicherheitseigenschaften erneut, weil die ursprüngliche
Herleitung nur den *Mittelpunkt* des Objekts, nicht seine *Oberfläche*
berücksichtigte: Bei größerem Radius reicht die Oberfläche weiter in Richtung
$x=0$ (in die Reset-Region) und weiter unter $z=0$ (durch den Boden) hinein.
Konkret gemessen mit dem alten Spawn-Bereich: minimaler Bodenabstand
$-6.3\,\mathrm{cm}$ (Durchdringung), minimaler Reset-Oberflächenabstand
$-4.7\,\mathrm{cm}$ (Überlappung).

**Fix:** Neuberechnung der Spawn-Region gegen den *maximalen* Radius statt
gegen einen Punktziel: $\varphi \in (-1.0, -0.6)$, $\rho \in (0.24, 0.32)$
(vorher $\varphi \in (-1.3, -0.3)$, $\rho \in (0.12, 0.22)$). Erneut über
2048 parallele Umgebungen verifiziert: minimaler Bodenabstand
$+4.2\,\mathrm{cm}$, minimaler Reset-Oberflächenabstand $+7.2\,\mathrm{cm}$.

**Methodische Lehre:** Eine geometrische Sicherheitsgarantie, die nur für
einen Parameterbereich hergeleitet und verifiziert wurde, überträgt sich nicht
automatisch auf eine Erweiterung dieses Bereichs (hier: des Radius) — sie muss
für den vollen neuen Parameterraum neu geprüft werden, im Zweifel empirisch
über die Randfälle der Parameterbox, nicht nur „von Hand“ für einen
vermeintlich ungünstigsten Fall.

### 5.6 Kraftverteilungs-Reward: Diagnose der Sensor-Dokumentationsdiskrepanz

Bei der Implementierung von `wrap_force_distribution` (Abschnitt 3.4) zeigte
ein erster Validierungsversuch scheinbar *keinerlei* Kontaktkraft, obwohl die
geometrischen Metriken (`min_surface_dist$\approx 1$–$2\,\mathrm{cm}$) auf engen
Kontakt hindeuteten. Zwei unabhängige Ursachen wurden identifiziert:

1. Die installierte mjlab-Version (aus dem `uv`-Lockfile) unterscheidet sich
   von der lokal eingebundenen Submodul-Referenz — die vom Nutzer bereitgestellte
   API-Dokumentation (`ContactSensor.primary_names`) existierte in der
   tatsächlich verwendeten Version noch nicht. Dies betraf nur ein
   Diagnoseskript, nicht die eigentliche Reward-Implementierung.
2. Im eigenen Validierungsskript fehlte ein `env.reset()`-Aufruf vor dem
   ersten Policy-Rollout, wodurch das Kommando nie initial gezogen wurde und
   das Objekt an seiner Kompilierungs-Default-Position (nahe dem Ursprung)
   verharrte, statt an der beabsichtigten Position.

Nach Korrektur beider Punkte zeigte sich das eigentliche, für die
Aufgabenstellung zentrale Ergebnis: Eine ausschließlich mit den geometrischen
Termen (3.2, 3.3) trainierte Policy erzeugte real gemessen im Mittel nur
$\approx 1.03$ von 14 Segmenten in tatsächlichem physischen Kontakt
(Gesamtkraft $\approx 43\,\mathrm{N}$, praktisch an einem einzigen Segment
konzentriert) — der Kern des vom Nutzer beschriebenen Problems, nun erstmals
mit realen Sensordaten belegt statt nur vermutet. Die Diskrepanz erklärt sich
daraus, dass `wrap_proximity`/`wrap_coverage` auf den `site_imu`-*Punkten* am
geometrischen Mittelpunkt jedes Segments rechnen, nicht auf dessen tatsächlicher
Quader-Kollisionsgeometrie — ein kleiner Site-Oberflächenabstand ist damit nur
eine Näherung für echten physischen Kontakt, keine Garantie dafür.

**Offener Befund:** Ein anschließender, kurzer Trainingslauf (150 Iterationen)
mit aktiviertem `wrap_force_distribution` zeigte den erwarteten Lernfortschritt
im geloggten Episodenreward des Terms selbst (monoton steigend von $0.0000$ auf
$0.0205$), *ohne* die geometrischen Metriken zu verschlechtern. Ein direkter
Vorher/Nachher-Vergleich über 512 Umgebungen (bester erreichter Wert pro
Umgebung über einen 600-Schritte-Rollout) ergab jedoch:

| | Kontakt überhaupt hergestellt | Segmente im Kontakt (Median) | `wrap_force_distribution` (Median) |
|---|---|---|---|
| vorher (nur geometrische Terme) | $100\,\%$ | 2.0 | 0.263 |
| nachher (150 Iter. mit Kraftterm) | $41\,\%$ | 1.0 | 0.000 |

Dieses Ergebnis ist **nicht** eindeutig positiv — die zweite (frisch
initialisierte, nicht vom ersten Checkpoint fortgesetzte) Trainingsinstanz
schnitt in diesem einzelnen Lauf schlechter ab. Da PPO-Trainingsläufe in
diesem Projekt wiederholt deutliche Lauf-zu-Lauf-Varianz bei identischer
Konfiguration zeigten (u. a. Abschnitte 5.1, 5.2), lässt sich aus einem
einzelnen Seed und 150 von konfigurierten 200 Iterationen keine belastbare
Aussage über die Wirksamkeit des neuen Terms ableiten. Es ist plausibel, dass
150 Iterationen nicht ausreichen, damit sich eine grundlegend andere
Greifstrategie (viele statt eines Segments) gegen die bereits eingespielten,
höher gewichteten geometrischen Terme durchsetzt.

---

## 6. Offene Fragen

Für die weitere Arbeit (und ggf. als Diskussion in der Masterarbeit) bleiben
folgende Punkte offen:

1. **Statistisch belastbare Bewertung von `wrap_force_distribution`.** Der in
   5.6 beschriebene Vergleich basiert auf einem einzelnen Seed und einer im
   Vergleich zur konfigurierten Trainingslänge kurzen Laufzeit. Für eine
   wissenschaftlich verwertbare Aussage wären mehrere Seeds über die volle
   Trainingslänge (200+ Iterationen) sowie ggf. ein Curriculum-Ansatz nötig,
   der den Kraftterm erst zuschaltet, nachdem die geometrischen Terme bereits
   greifen (analog zum bestehenden `dr_range_curriculum`, Abschnitt 1.5).
2. **Geometrische vs. physikalische Kontaktapproximation.** Die
   in 5.6 aufgedeckte Lücke zwischen site-basierter Abstandsnäherung und realer
   Kollisionsgeometrie besteht für `wrap_proximity`/`wrap_coverage` weiterhin.
   Eine geometrisch exaktere Distanzmessung (z. B. gegen die tatsächliche
   Quaderoberfläche statt gegen einen Punkt) könnte die Notwendigkeit des
   Kraftterms verringern oder dessen Zusammenspiel mit den geometrischen Termen
   verbessern.
3. **Fehlende Mindestgriffkraft.** `wrap_force_distribution` ist bewusst
   skaleninvariant (Abschnitt 3.4) und bewertet daher einen sehr leichten,
   gleichmäßigen Kontakt genauso gut wie einen festen. Ob das für die
   angestrebte Anwendung (z. B. spätere Übertragung auf reale Hardware)
   ausreicht oder ein zusätzlicher, absoluter Kraftterm nötig ist, ist offen.
4. **Kein Kontaktsensor-Feedback in der Beobachtung.** Der `ContactSensor`
   wird aktuell ausschließlich für den Reward genutzt, nicht als
   Policy-Beobachtung. Ob das Hinzufügen von (ggf. auf die jeweilige
   `SensorLevel`-Stufe angepasster) Kontaktinformation als Beobachtung das
   Lernen beschleunigt, wurde nicht untersucht.
