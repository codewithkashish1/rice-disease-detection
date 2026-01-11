import pandas as pd

class MedicineService:
    def __init__(self, csv_path="datasets/medicine.csv"):
        self.df = pd.read_csv(csv_path)

    def get_medicine(self, crop, predicted_label):
        """
        Handles labels like:
        - Rice_Blast
        - Pepper__bell___Bacterial_spot
        """

        if not predicted_label:
            return None

        label = predicted_label.lower()

        # -------------------------------
        # HANDLE PEPPER LABELS
        # -------------------------------
        if "pepper" in label:
            crop_clean = "pepper"
            disease_clean = label.split("___")[-1]

        # -------------------------------
        # HANDLE RICE / PULSES
        # -------------------------------
        else:
            crop_clean = crop.lower()
            disease_clean = label

        disease_clean = disease_clean.replace("_", " ").strip()

        # -------------------------------
        # MATCH WITH CSV
        # -------------------------------
        for _, row in self.df.iterrows():
            csv_crop = str(row["crop"]).strip().lower()
            csv_disease = str(row["disease"]).strip().lower().replace("_", " ")

            if csv_crop == crop_clean and csv_disease == disease_clean:
                return row.to_dict()

        return None
