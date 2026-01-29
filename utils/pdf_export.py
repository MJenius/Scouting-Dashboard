"""
pdf_export.py - Generate a one-page scouting dossier PDF for a player using fpdf.
"""

from fpdf import FPDF
import pandas as pd
from typing import Dict

class ScoutingDossierPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'One-Page Scouting Dossier', ln=True, align='C')
        self.ln(5)

    def add_player_info(self, player_data: pd.Series):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, f"{player_data.get('Player', 'Unknown')} ({player_data.get('Primary_Pos', '-')})", ln=True)
        self.set_font('Arial', '', 11)
        self.cell(0, 8, f"Age: {player_data.get('Age', '-')}", ln=True)
        self.cell(0, 8, f"Squad: {player_data.get('Squad', '-')}", ln=True)
        self.cell(0, 8, f"League: {player_data.get('League', '-')}", ln=True)
        self.cell(0, 8, f"Archetype: {player_data.get('Archetype', '-')}", ln=True)
        self.ln(2)

    def add_key_stats(self, player_data: pd.Series):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Key Statistics', ln=True)
        self.set_font('Arial', '', 11)
        for feat in ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'Crs/90', 'Int/90', 'TklW/90']:
            value = player_data.get(feat, '-')
            pct = player_data.get(f'{feat}_pct', '-')
            self.cell(0, 8, f"{feat}: {value}  (Percentile: {pct})", ln=True)
        self.ln(2)

    def add_narrative(self, narrative: str):
        self.set_font('Arial', 'I', 11)
        self.multi_cell(0, 8, narrative)
        self.ln(2)

    def add_market_value(self, player_data: pd.Series):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Market Value', ln=True)
        self.set_font('Arial', '', 11)
        value = player_data.get('Estimated_Value_£M', '-')
        tier = player_data.get('Value_Tier', '-')
        self.cell(0, 8, f"Estimated Value: £{value}M", ln=True)
        self.cell(0, 8, f"Value Tier: {tier}", ln=True)
        self.ln(2)

    def add_confidence(self, completeness: float):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Scouting Confidence', ln=True)
        self.set_font('Arial', '', 11)
        if completeness >= 90:
            label = "Verified Elite Data"
        elif completeness >= 70:
            label = "Good Scouting Data"
        elif completeness >= 40:
            label = "Directional Data (Further Vetting Required)"
        else:
            label = "Incomplete Data - Caution"
        self.cell(0, 8, f"{label} ({completeness:.0f}%)", ln=True)
        self.ln(2)


def export_scouting_pdf(player_data: pd.Series, narrative: str, file_path: str):
    pdf = ScoutingDossierPDF()
    pdf.add_page()
    pdf.add_player_info(player_data)
    pdf.add_key_stats(player_data)
    pdf.add_market_value(player_data)
    pdf.add_confidence(player_data.get('Completeness_Score', 0))
    pdf.add_narrative(narrative)
    pdf.output(file_path)
