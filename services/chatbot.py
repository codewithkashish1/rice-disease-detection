class CropChatbot:
    def get_response(self, user_input):
        user_input = user_input.lower()

        # ==================================================
        # PRECAUTIONS / MANAGEMENT / TREATMENT (CHECK FIRST)
        # ==================================================
        if any(word in user_input for word in [
            "precaution", "precautions", "management", "manage",
            "treatment", "control", "protect", "protection", "tips"
        ]):

            # ---------- Disease-specific management ----------
            if "rice blast" in user_input or "blast" in user_input:
                return (
                    "**Precautions & Management for Rice Blast:**\n"
                    "- Use resistant rice varieties\n"
                    "- Avoid dense planting\n"
                    "- Maintain proper drainage\n"
                    "- Remove infected plant debris\n"
                    "- Apply recommended fungicides"
                )

            elif "tungro" in user_input:
                return (
                    "**Precautions & Management for Tungro:**\n"
                    "- Use resistant varieties\n"
                    "- Control insect vectors (leafhoppers)\n"
                    "- Remove infected plants early\n"
                    "- Avoid staggered planting"
                )

            elif "anthracnose" in user_input:
                return (
                    "**Precautions & Management for Anthracnose:**\n"
                    "- Crop rotation\n"
                    "- Remove infected plant debris\n"
                    "- Use disease-free seeds\n"
                    "- Apply fungicides if required"
                )

            elif "bacterial spot" in user_input:
                return (
                    "**Precautions & Management for Bacterial Spot:**\n"
                    "- Use certified disease-free seeds\n"
                    "- Avoid overhead irrigation\n"
                    "- Remove infected leaves\n"
                    "- Apply copper-based bactericides"
                )

            # ---------- GENERAL management (IMPORTANT) ----------
            else:
                return (
                    "**General Disease Management for Rice & Pulses:**\n"
                    "- Early disease detection and monitoring\n"
                    "- Use resistant crop varieties\n"
                    "- Maintain proper plant spacing\n"
                    "- Avoid waterlogging\n"
                    "- Field sanitation\n"
                    "- Timely use of fungicides/pesticides"
                )

        # =======================
        # RICE DISEASE INFORMATION
        # =======================
        elif "rice blast" in user_input or "blast" in user_input:
            return (
                "**Rice Blast**\n"
                "- Description: A fungal disease affecting rice leaves.\n"
                "- Symptoms: Gray or white lesions with brown margins."
            )

        elif "tungro" in user_input:
            return (
                "**Tungro Disease**\n"
                "- Description: Viral disease of rice spread by leafhoppers.\n"
                "- Symptoms: Yellow-orange leaves and stunted growth."
            )

        elif "rice" in user_input:
            return (
                "**Rice Diseases:**\n"
                "- Rice Blast\n"
                "- Tungro\n\n"
                "Ask a disease name or its precautions."
            )

        # =========================
        # PULSES DISEASE INFORMATION
        # =========================
        elif "anthracnose" in user_input:
            return (
                "**Anthracnose (Pulses)**\n"
                "- Description: Fungal disease affecting pulses.\n"
                "- Symptoms: Dark sunken lesions on leaves and stems."
            )

        elif "bacterial spot" in user_input:
            return (
                "**Bacterial Spot (Pulses)**\n"
                "- Description: Bacterial infection of leaves.\n"
                "- Symptoms: Water-soaked spots turning brown."
            )

        elif "pulses" in user_input or "pepper" in user_input:
            return (
                "**Pulses Diseases:**\n"
                "- Bacterial Spot\n"
                "- Anthracnose\n\n"
                "Ask a disease name or its precautions."
            )

        # =========
        # SUMMARY
        # =========
        elif "summary" in user_input:
            return (
                "**Disease Summary:**\n"
                "- Rice: Rice Blast, Tungro\n"
                "- Pulses: Bacterial Spot, Anthracnose"
            )

        # =========
        # FALLBACK
        # =========
        else:
            return (
                "Sorry, I didn't understand.\n"
                "You can ask about:\n"
                "- Rice or pulses diseases\n"
                "- Disease precautions or management\n"
                "- Disease summary"
            )
