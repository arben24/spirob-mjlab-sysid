import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

class SpirobKinematics:
    """
    Diese Klasse bildet die Kinematik des Spirob-Roboters ab und 
    beinhaltet den Optimierungs-Ansatz (Methode 3) zur Winkelberechnung.
    
    Optimiert für:
    1. Scharniergelenke (Nur 1 Gelenkwinkel pro Gelenk um die Y-Achse).
    2. Bewegliche Basis (KUKA-Arm): Der Basissensor (Segment 0) dient als dynamische Referenz.
    3. Dynamische Sensoranzahl: Passt sich automatisch der Anzahl der übergebenen Sensor-IDs an.
    """
    def __init__(self, sensor_id_mapping=None, rotation_axis='x'):
        # Wenn kein Mapping übergeben wird, nutzen wir die aktuell erkannten 14 Sensoren
        # sortiert von klein (Basis) nach groß (Spitze):
        # 0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37
        if sensor_id_mapping is None:
            self.sensor_id_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37]
        else:
            self.sensor_id_mapping = sensor_id_mapping
            
        self.rotation_axis = rotation_axis.lower()
        
        # Dynamische Skalierung basierend auf der Anzahl der Sensoren
        self.num_segments = len(self.sensor_id_mapping)
        self.num_joints = self.num_segments - 1 # N Sensoren können N-1 Gelenkwinkel bestimmen
        
        self.max_angle_deg = 30.0
        self.max_angle_rad = np.radians(self.max_angle_deg)

    def forward_kinematics(self, q, measured_base_acc):
        """
        Berechnet die theoretisch zu erwartenden Sensorwerte für gegebene Winkel.
        Die Basis wird dynamisch anhand des gemessenen Vektors der Basis ausgerichtet.
        
        Args:
            q: 1D Array mit self.num_joints Werten (Gelenkwinkel in Radiant)
            measured_base_acc: 3D-Vektor [x,y,z] des Sensors auf der Basis (bereits normiert!)
        Returns:
            expected_acc: (num_segments)x3 Array der simulierten Acc-Werte
        """
        expected_acc = np.zeros((self.num_segments, 3))
        
        # Die Basis (Segment 0) hat als Orientierung genau den gemessenen Schwerkraftvektor.
        # Damit ist der Fehler an Segment 0 definitionsgemäß immer 0.
        expected_acc[0] = measured_base_acc
        
        # Start der Rotationskette relativ zur Basis (Identität)
        R_abs = R.from_matrix(np.eye(3))
        
        for i in range(self.num_joints):
            angle = q[i] # Gelenkwinkel
            
            # Relative Rotation des Scharniers
            R_joint = R.from_euler(self.rotation_axis, angle)
            R_abs = R_abs * R_joint
            
            # Projiziere den gemessenen Schwerkraftvektor der Basis in das lokale Segment
            acc_local = R_abs.inv().apply(measured_base_acc)
            expected_acc[i+1] = acc_local
            
        return expected_acc
        
    def cost_function(self, q, measured_acc, q_prev=None, regularization_weight=0.0):
        """
        Die Kostenfunktion berechnet den Fehler zwischen Simulation und Messung.
        measured_acc: (num_segments)x3 Matrix, sortiert in physischer Reihenfolge.
        """
        # Der erste Eintrag ist die gemessene Beschleunigung der Basis (Segment 0)
        measured_base_acc = measured_acc[0]
        
        expected_acc = self.forward_kinematics(q, measured_base_acc)
        residuals = expected_acc.flatten() - measured_acc.flatten()
        
        if q_prev is not None and regularization_weight > 0.0:
            reg = np.sqrt(regularization_weight) * (q - q_prev)
            residuals = np.concatenate([residuals, reg])
            
        return residuals
    
    def prepare_measured_data(self, raw_sensor_dict):
        """
        Hilfsfunktion: Sortiert die echten Sensordaten basierend auf dem Mapping 
        in die physische Reihenfolge (Basis -> Spitze) und normiert die Vektoren.
        
        Args:
            raw_sensor_dict: Dictionary mit {sensor_id: [acc_x, acc_y, acc_z]}
        Returns:
            ordered_normalized_acc: (num_segments)x3 Array bereit für den Solver
        """
        ordered_acc = np.zeros((self.num_segments, 3))
        
        for segment_idx, sensor_id in enumerate(self.sensor_id_mapping):
            if sensor_id in raw_sensor_dict:
                ordered_acc[segment_idx] = raw_sensor_dict[sensor_id]
            else:
                # Fallback, falls ein Sensor ausfällt
                print(f"WARNUNG: Sensor ID {sensor_id} für Segment {segment_idx} fehlt!")
                ordered_acc[segment_idx] = [0.0, 0.0, 1.0]
                
        # Normierung aller Vektoren auf die Länge 1.0
        lengths = np.linalg.norm(ordered_acc, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0 
        
        return ordered_acc / lengths

    def solve_angles_direct(self, measured_acc, clip=True):
        """
        Geschlossene Winkelbestimmung ohne Iteration (atan2-Ablesung).

        Da alle Gelenke um dieselbe Achse drehen, bleibt die Komponente entlang
        der Drehachse unveraendert und die gesamte Drehung findet in der Ebene
        senkrecht dazu statt. Der Richtungswinkel des gemessenen Vektors in
        dieser Ebene entspricht der absoluten Neigung des Segments; die
        Differenz benachbarter Segmente ergibt den Gelenkwinkel.

        Hinweis: Im reinen ACC-Modell zerfaellt das Least-Squares-Problem in
        unabhaengige Ein-Parameter-Probleme, die diese Ablesung exakt loest.
        Das Ergebnis stimmt daher (ohne aktive Grenzen) mit solve_angles ueberein.

        Args:
            measured_acc: (num_segments)x3 Array, sortiert und normiert.
            clip: Ergebnis auf die mechanischen Grenzen begrenzen.
        Returns:
            q: 1D Array mit num_joints Gelenkwinkeln in Radiant.
        """
        a = np.asarray(measured_acc, dtype=float)

        # Ebene senkrecht zur Drehachse: (erste Achse, zweite Achse)
        if self.rotation_axis == 'x':
            first, second = 1, 2   # y, z
        elif self.rotation_axis == 'y':
            first, second = 2, 0   # z, x
        elif self.rotation_axis == 'z':
            first, second = 0, 1   # x, y
        else:
            raise ValueError(f"Unbekannte Rotationsachse: {self.rotation_axis}")

        phi = np.arctan2(a[:, second], a[:, first])

        # Gelenkwinkel als Differenz benachbarter Richtungswinkel,
        # auf (-pi, pi] normiert, damit der Sprung bei +-180 Grad nicht stoert.
        q = phi[:-1] - phi[1:]
        q = (q + np.pi) % (2.0 * np.pi) - np.pi

        if clip:
            q = np.clip(q, -self.max_angle_rad, self.max_angle_rad)

        return q

    def cost_of(self, q, measured_acc):
        """Kostenwert in der Definition von SciPy (0.5 * Summe der Residuenquadrate)."""
        res = self.cost_function(q, measured_acc)
        return 0.5 * float(np.sum(res ** 2))

    def jacobian_function(self, q, measured_acc, q_prev=None, regularization_weight=0.0):
        """
        Berechnet die analytische Jacobi-Matrix der Residuen bezüglich der Gelenkwinkel q.
        Unterstützt X, Y und Z als Rotationsachse.
        """
        num_segments = self.num_segments
        num_joints = self.num_joints
        
        # Vorwärtskinematik-Durchlauf für die erwarteten Beschleunigungen
        measured_base_acc = measured_acc[0]
        expected_acc = self.forward_kinematics(q, measured_base_acc)
        
        # D[k, j, m] = Ableitung von Segment k, Koordinate m bzgl. Gelenkwinkel j
        D = np.zeros((num_segments, num_joints, 3))
        
        for k in range(1, num_segments):
            theta = q[k-1]
            c, s = np.cos(theta), np.sin(theta)
            
            # Diagonalterm (j = k-1)
            a_prev = expected_acc[k-1]
            ax, ay, az = a_prev[0], a_prev[1], a_prev[2]
            
            if self.rotation_axis == 'x':
                D[k, k-1, 0] = 0.0
                D[k, k-1, 1] = -ay * s + az * c
                D[k, k-1, 2] = -ay * c - az * s
            elif self.rotation_axis == 'y':
                D[k, k-1, 0] = -ax * s - az * c
                D[k, k-1, 1] = 0.0
                D[k, k-1, 2] = ax * c - az * s
            elif self.rotation_axis == 'z':
                D[k, k-1, 0] = -ax * s + ay * c
                D[k, k-1, 1] = -ax * c - ay * s
                D[k, k-1, 2] = 0.0
                
            # Off-Diagonalterme (j < k-1)
            if k > 1:
                dx = D[k-1, :k-1, 0]
                dy = D[k-1, :k-1, 1]
                dz = D[k-1, :k-1, 2]
                
                if self.rotation_axis == 'x':
                    D[k, :k-1, 0] = dx
                    D[k, :k-1, 1] = dy * c + dz * s
                    D[k, :k-1, 2] = -dy * s + dz * c
                elif self.rotation_axis == 'y':
                    D[k, :k-1, 0] = dx * c - dz * s
                    D[k, :k-1, 1] = dy
                    D[k, :k-1, 2] = dx * s + dz * c
                elif self.rotation_axis == 'z':
                    D[k, :k-1, 0] = dx * c + dy * s
                    D[k, :k-1, 1] = -dx * s + dy * c
                    D[k, :k-1, 2] = dz
                    
        # Reshape zu (3 * num_segments, num_joints)
        J = D.transpose(0, 2, 1).reshape(3 * num_segments, num_joints)
        
        if q_prev is not None and regularization_weight > 0.0:
            reg_jac = np.sqrt(regularization_weight) * np.eye(num_joints)
            J = np.vstack([J, reg_jac])
            
        return J

    def solve_angles(self, measured_acc, q0=None, q_prev=None, regularization_weight=0.0):
        """
        Sucht die Gelenkwinkel für ein bereits sortiertes und normiertes Array.
        """
        if q0 is None:
            q0 = np.zeros(self.num_joints)
        lower_bounds = [-self.max_angle_rad] * self.num_joints
        upper_bounds = [self.max_angle_rad] * self.num_joints
        bounds = (lower_bounds, upper_bounds)
        
        result = least_squares(
            self.cost_function, 
            x0=q0, 
            args=(measured_acc, q_prev, regularization_weight), 
            bounds=bounds,
            jac=self.jacobian_function,
            method='trf', 
            ftol=1e-6,
            xtol=1e-6
        )
        
        return result.x, result.success, result.cost

# ==========================================
# Testlauf mit den echten 13 Sensoren
# ==========================================
if __name__ == "__main__":
    # Wir initialisieren die Klasse mit dem Standard-Mapping (14 Sensoren)
    kin = SpirobKinematics()
    
    print(f"=== Testlauf (14 Sensoren aktiv, Gelenke zu lösen: {kin.num_joints}) ===\n")
    print("Sensor ID Mapping (Basis -> Spitze):")
    print(kin.sensor_id_mapping)
    print("")
    
    # 1. Schiefstehende Basis simulieren (KUKA-Neigung)
    true_base_acc = np.array([0.5, 0.0, 0.866]) # ca. 30° geneigt
    
    # 2. Wahre Gelenkwinkel der 13 Gelenke (in Grad)
    true_q = np.zeros(kin.num_joints)
    true_q[0] = np.radians(20)   # Erstes Gelenk biegt um 20°
    true_q[5] = np.radians(-15)  # Gelenk 5 biegt um -15°
    true_q[12] = np.radians(30)  # Gelenk 12 (letztes messbares Gelenk) biegt um 30°
    
    print("Wahre Winkel des Spirobs relativ zur Basis (Grad):")
    print(np.round(np.degrees(true_q), 1))
    
    # 3. Perfekte Sensordaten im Raum berechnen (physische Reihenfolge)
    perfect_acc_physical = kin.forward_kinematics(true_q, true_base_acc)
    
    # 4. In ungeordnetes Hardware-Format packen (wie es aus dem Serial-Port purzelt)
    raw_hardware_data = {}
    for segment_idx, sensor_id in enumerate(kin.sensor_id_mapping):
        raw_hardware_data[sensor_id] = perfect_acc_physical[segment_idx]
        
    # 5. Rauschen hinzufügen (2% Störsignal)
    np.random.seed(42)
    for sensor_id in raw_hardware_data:
        noise = np.random.normal(0, 0.02, 3)
        raw_hardware_data[sensor_id] = raw_hardware_data[sensor_id] + noise

    # 6. Daten vorbereiten (Sortieren & Normieren)
    prepared_acc = kin.prepare_measured_data(raw_hardware_data)
    
    # 7. Optimierung ausführen
    print("\nStarte Optimierer...")
    estimated_q, success, final_cost = kin.solve_angles(prepared_acc)
    
    print(f"Erfolgreich? {success} (Finale Kosten: {final_cost:.6f})")
    print("\nGeschätzte Winkel relativ zur geneigten Basis (Grad):")
    print(np.round(np.degrees(estimated_q), 1))
