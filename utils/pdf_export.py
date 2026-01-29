"""
pdf_export.py - Generate a one-page scouting dossier PDF for a player.

Enhanced version with:
- Radar chart image embedding
- Professional layout
- Market value and confidence scoring
- AI-generated narratives
"""

from fpdf import FPDF
import pandas as pd
import os
import tempfile
from typing import Dict, Optional
import re


def sanitize_text_for_pdf(text: str) -> str:
    """
    Remove Unicode characters that aren't supported by standard PDF fonts.
    
    Args:
        text: Input text that may contain emojis or special Unicode characters
        
    Returns:
        Sanitized text with only ASCII-compatible characters
    """
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '➖': '-',
        '—': '-',
        '–': '-',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        '…': '...',
        '•': '*',
        '→': '->',
        '←': '<-',
        '✓': '[OK]',
        '✗': '[X]',
        '⚠': '[!]',
        '⚽': '',
        '🌟': '*',
        '💎': '',
        '📊': '',
        '🔍': '',
        '⚔️': '',
        '📥': '',
    }
    
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text


class ScoutingDossierPDF(FPDF):
    """Enhanced PDF generator for scouting dossiers."""
    
    def header(self):
        """PDF header with title."""
        self.set_font('Arial', 'B', 18)
        self.cell(0, 12, 'SCOUTING DOSSIER', ln=True, align='C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 6, 'Professional Player Analysis Report', ln=True, align='C')
        self.ln(5)

    def add_player_info(self, player_data: pd.Series):
        """Add player basic information section."""
        self.set_font('Arial', 'B', 14)
        player_name = sanitize_text_for_pdf(str(player_data.get('Player', 'Unknown Player')))
        self.cell(0, 10, player_name, ln=True)
        
        self.set_font('Arial', '', 11)
        info_lines = [
            f"Position: {player_data.get('Primary_Pos', '-')}  |  Age: {player_data.get('Age', '-')}  |  90s Played: {player_data.get('90s', '-')}",
            f"Club: {sanitize_text_for_pdf(str(player_data.get('Squad', '-')))}",
            f"League: {sanitize_text_for_pdf(str(player_data.get('League', '-')))}  (Tier {player_data.get('League_Tier', '-')})",
            f"Archetype: {sanitize_text_for_pdf(str(player_data.get('Archetype', '-')))}",
        ]
        
        for line in info_lines:
            self.cell(0, 6, line, ln=True)
        
        self.ln(3)

    def add_key_stats(self, player_data: pd.Series):
        """Add key statistics section."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, 'KEY STATISTICS (Per 90 Minutes)', ln=True)
        
        self.set_font('Arial', '', 10)
        
        # Determine which stats to show based on position
        position = player_data.get('Primary_Pos', '')
        
        if position == 'GK':
            stats = ['GA90', 'Save%', 'CS%', 'Saves']
        else:
            stats = ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'Crs/90', 'Int/90', 'TklW/90']
        
        for feat in stats:
            if feat in player_data.index:
                value = player_data.get(feat, '-')
                pct_col = f'{feat}_pct'
                pct = player_data.get(pct_col, '-')
                
                # Format value
                if isinstance(value, (int, float)):
                    value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                
                # Format percentile
                if isinstance(pct, (int, float)):
                    pct_str = f"{pct:.0f}%"
                else:
                    pct_str = "-"
                
                self.cell(0, 5, f"  {feat}: {value_str}  (Percentile: {pct_str})", ln=True)
        
        self.ln(2)

    def add_radar_chart(self, image_path: str):
        """Embed radar chart image."""
        if os.path.exists(image_path):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 8, 'PERFORMANCE PROFILE', ln=True)
            
            # Add image (centered, scaled to fit)
            self.image(image_path, x=30, w=150)
            self.ln(5)

    def add_narrative(self, narrative: str):
        """Add scout's narrative section."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, "SCOUT'S TAKE", ln=True)
        
        self.set_font('Arial', '', 10)
        # Sanitize narrative to remove unsupported Unicode characters
        clean_narrative = sanitize_text_for_pdf(narrative)
        self.multi_cell(0, 5, clean_narrative)
        self.ln(2)

    def add_market_value(self, player_data: pd.Series):
        """Add market value section."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, 'MARKET VALUATION', ln=True)
        
        self.set_font('Arial', '', 10)
        value = player_data.get('Estimated_Value_£M', '-')
        tier = player_data.get('Value_Tier', '-')
        
        if isinstance(value, (int, float)):
            value_str = f"£{value:.1f}M"
        else:
            value_str = str(value)
        
        self.cell(0, 6, f"Estimated Value: {value_str}", ln=True)
        self.cell(0, 6, f"Value Tier: {tier}", ln=True)
        self.ln(2)

    def add_confidence(self, completeness: float):
        """Add data confidence section."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, 'DATA CONFIDENCE', ln=True)
        
        self.set_font('Arial', '', 10)
        
        if completeness >= 90:
            label = "[VERIFIED] Elite Data"
        elif completeness >= 70:
            label = "[GOOD] Scouting Data"
        elif completeness >= 40:
            label = "[CAUTION] Directional Data (Further Vetting Required)"
        else:
            label = "[WARNING] Incomplete Data - Caution Advised"
        
        self.cell(0, 6, f"{label} ({completeness:.0f}% complete)", ln=True)
        self.ln(2)


def save_radar_chart_image(
    player_data: pd.Series,
    comparison_data: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate and save radar chart as PNG image.
    
    Args:
        player_data: Player statistics
        comparison_data: Optional comparison player data
        save_path: Path to save image (defaults to temp file)
        
    Returns:
        Path to saved image file
    """
    from utils.similarity import RadarChartGenerator
    
    if save_path is None:
        # Create temp file
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, f"radar_{player_data.get('Player', 'player').replace(' ', '_')}.png")
    
    # Get player profile
    from utils.constants import FEATURE_COLUMNS
    
    target_stats = {feat: player_data.get(feat, 0) for feat in FEATURE_COLUMNS if feat in player_data.index}
    
    comparison_stats = None
    if comparison_data is not None:
        comparison_stats = {feat: comparison_data.get(feat, 0) for feat in FEATURE_COLUMNS if feat in comparison_data.index}
    
    # Generate radar chart
    generator = RadarChartGenerator()
    generator.generate_matplotlib_radar(
        target_stats=target_stats,
        comparison_stats=comparison_stats,
        target_name=player_data.get('Player', 'Player'),
        comparison_name=comparison_data.get('Player', 'Comparison') if comparison_data is not None else '',
        save_path=save_path,
        dpi=200,
    )
    
    return save_path


def create_recruitment_dossier(
    player_data: pd.Series,
    narrative: str,
    output_path: str,
    radar_image_path: Optional[str] = None,
) -> str:
    """
    Create comprehensive one-page recruitment dossier PDF.
    
    Args:
        player_data: Player statistics
        narrative: Scout's narrative text
        output_path: Path to save PDF
        radar_image_path: Optional path to radar chart image (will generate if not provided)
        
    Returns:
        Path to created PDF file
    """
    # Generate radar chart if not provided
    if radar_image_path is None or not os.path.exists(radar_image_path):
        radar_image_path = save_radar_chart_image(player_data)
    
    # Create PDF
    pdf = ScoutingDossierPDF()
    pdf.add_page()
    
    # Add sections
    pdf.add_player_info(player_data)
    pdf.add_key_stats(player_data)
    pdf.add_radar_chart(radar_image_path)
    pdf.add_market_value(player_data)
    pdf.add_confidence(player_data.get('Completeness_Score', 0))
    pdf.add_narrative(narrative)
    
    # Save PDF
    pdf.output(output_path)
    
    return output_path


def export_scouting_pdf(
    player_data: pd.Series,
    narrative: str,
    file_path: str,
) -> str:
    """
    Convenience function to export scouting dossier PDF.
    
    Args:
        player_data: Player statistics
        narrative: Scout's narrative
        file_path: Output PDF path
        
    Returns:
        Path to created PDF
    """
    return create_recruitment_dossier(player_data, narrative, file_path)
