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
import unicodedata
from .constants import (
    FEATURE_COLUMNS, 
    GK_FEATURE_COLUMNS, 
    LEAGUE_TIERS,
    RADAR_LABELS
)


def sanitize_text_for_pdf(text: str) -> str:
    """
    Ultiamte defensive sanitization to stop PDF font crashes.
    """
    if text is None:
        return "-"
    if not isinstance(text, str):
        text = str(text)
        
    # Standardize unicode characters
    text = unicodedata.normalize('NFKD', text)
        
    # Manual replacements for common football symbols that we want to keep as text
    # Using hex codes for absolute certainty
    replacements = {
        '\u2796': '-', # ➖ Heavy Minus
        '\u2212': '-', # − Minus
        '\u2013': '-', # – En Dash
        '\u2014': '-', # — Em Dash
        '\u2015': '-', # ― Horizontal Bar
        '\u2018': "'", # ‘
        '\u2019': "'", # ’
        '\u201c': '"', # “
        '\u201d': '"', # ”
        '\u2026': '...', # …
        '\u2022': '*', # •
        '\u2192': '->', # →
        '\u2191': '^', # ↑ 
        '\u2193': 'v', # ↓
        '\u2713': '[OK]', # ✓
        '\u2717': '[X]', # ✗
        '\u26a0': '[!]', # ⚠
        '\xa3': 'GBP ', # £
    }
    
    for uni, asc in replacements.items():
        text = text.replace(uni, asc)
    
    # Final safety: drop ANY remaining non-ASCII character
    # Standard PDF fonts (Helvetica/Arial) will crash on anything > 127
    return "".join(c if ord(c) < 128 else "" for c in text)


class ScoutingDossierPDF(FPDF):
    """Enhanced PDF generator for scouting dossiers."""
    
    def header(self):
        """PDF header with title."""
        self.set_font('Arial', 'B', 18)
        self.cell(0, 12, sanitize_text_for_pdf('SCOUTING DOSSIER'), ln=True, align='C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 6, sanitize_text_for_pdf('Professional Player Analysis Report'), ln=True, align='C')
        self.ln(5)

    def add_player_info(self, player_data: pd.Series):
        """Add player basic information section."""
        self.set_font('Arial', 'B', 14)
        player_name = sanitize_text_for_pdf(player_data.get('Player', 'Unknown Player'))
        self.cell(0, 10, player_name, ln=True)
        
        self.set_font('Arial', '', 11)
        
        # Format strings then sanitize
        pos = player_data.get('Primary_Pos', '-')
        age = player_data.get('Age', '-')
        nineties = player_data.get('90s', '-')
        club = player_data.get('Squad', '-')
        league = player_data.get('League', '-')
        
        # Get tier from constants if missing in data
        tier = player_data.get('League_Tier')
        if pd.isna(tier) or tier == '-':
            tier = LEAGUE_TIERS.get(league, '-')
            
        arch = player_data.get('Archetype', '-')
        
        l1 = sanitize_text_for_pdf(f"Position: {pos}  |  Age: {age}  |  90s Played: {nineties}")
        l2 = sanitize_text_for_pdf(f"Club: {club}")
        l3 = sanitize_text_for_pdf(f"League: {league}  (Tier {tier})")
        l4 = sanitize_text_for_pdf(f"Archetype: {arch}")
        
        for line in [l1, l2, l3, l4]:
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
            # Show core plus advanced metrics for accuracy
            stats = [
                'Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 
                'xG90', 'xA90', 'xGChain90', 'xGBuildup90',
                'Int/90', 'TklW/90', 'Crs/90'
            ]
        
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
                
                # Sanitize the entire cell content
                stat_line = sanitize_text_for_pdf(f"  {feat}: {value_str}  (Percentile: {pct_str})")
                self.cell(0, 5, stat_line, ln=True)
        
        self.ln(2)

    def add_radar_chart(self, image_path: str):
        """Embed radar chart image."""
        if os.path.exists(image_path):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 8, sanitize_text_for_pdf('PERFORMANCE PROFILE'), ln=True)
            
            # Add image (centered, scaled to fit)
            self.image(image_path, x=30, w=150)
            self.ln(5)

    def add_narrative(self, narrative: str):
        """Add scout's narrative section."""
        self.set_font('Arial', 'B', 12)
        # Ensure header is sanitized
        self.cell(0, 8, sanitize_text_for_pdf("SCOUT'S TAKE"), ln=True)
        
        self.set_font('Arial', '', 10)
        # Strip Markdown bold/italic
        clean_narrative = narrative.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
        # Deep sanitize
        clean_narrative = sanitize_text_for_pdf(clean_narrative)
        self.multi_cell(0, 5, clean_narrative)
        self.ln(2)

    def add_market_value(self, player_data: pd.Series):
        """Add market value section."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, sanitize_text_for_pdf('MARKET VALUATION'), ln=True)
        
        self.set_font('Arial', '', 10)
        value = player_data.get('Estimated_Value_£M', '-')
        tier = player_data.get('Value_Tier', '-')
        
        if isinstance(value, (int, float)):
            value_str = f"£{value:.1f}M"
        else:
            value_str = str(value)
        
        line1 = sanitize_text_for_pdf(f"Estimated Value: {value_str}")
        line2 = sanitize_text_for_pdf(f"Value Tier: {tier}")
        
        self.cell(0, 6, line1, ln=True)
        self.cell(0, 6, line2, ln=True)
        self.ln(2)

    def add_confidence(self, completeness: float):
        """Add data confidence section."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, sanitize_text_for_pdf('DATA CONFIDENCE'), ln=True)
        
        self.set_font('Arial', '', 10)
        
        if completeness >= 90:
            label = "[VERIFIED] Elite Data"
        elif completeness >= 70:
            label = "[GOOD] Scouting Data"
        elif completeness >= 40:
            label = "[CAUTION] Directional Data (Further Vetting Required)"
        else:
            label = "[WARNING] Incomplete Data - Caution Advised"
        
        confidence_line = sanitize_text_for_pdf(f"{label} ({completeness:.0f}% complete)")
        self.cell(0, 6, confidence_line, ln=True)
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
    
    # Get player profile - ALWAYS use percentiles for the PDF radar chart to ensure visual scale (0-100)
    # The raw per-90 stats are too small to visualize on a 0-100 scale.
    from utils.constants import FEATURE_COLUMNS, GK_FEATURE_COLUMNS
    
    is_gk = player_data.get('Primary_Pos') == 'GK'
    feats = GK_FEATURE_COLUMNS if is_gk else FEATURE_COLUMNS
    
    # Extract percentiles for the radar
    target_stats = {}
    for feat in feats:
        pct_col = f"{feat}_pct"
        target_stats[feat] = player_data.get(pct_col, 0)
    
    comparison_stats = None
    if comparison_data is not None:
        comparison_stats = {}
        for feat in feats:
            pct_col = f"{feat}_pct"
            comparison_stats[feat] = comparison_data.get(pct_col, 0)
    
    # Generate radar chart
    generator = RadarChartGenerator()
    # Note: generator already handles labels from RADAR_LABELS
    generator.generate_matplotlib_radar(
        target_stats=target_stats,
        comparison_stats=comparison_stats,
        target_name=player_data.get('Player', 'Player'),
        comparison_name=comparison_data.get('Player', 'Comparison') if comparison_data is not None else '',
        save_path=save_path,
        dpi=150, # Lower DPI slightly for faster PDF generation and smaller file size
    )
    
    return save_path


def generate_dossier(
    player_data: pd.Series,
    narrative: str,
    output_path: str,
    radar_image_path: Optional[str] = None,
) -> str:
    """
    Fully implemented recruitment dossier generation.
    Includes Player Card, Radar Chart, and Scout's Take.
    """
    # Generate radar chart if not provided
    if radar_image_path is None or not os.path.exists(radar_image_path):
        radar_image_path = save_radar_chart_image(player_data)
    
    # Create PDF
    pdf = ScoutingDossierPDF()
    pdf.add_page()
    
    # Add sections in professional order
    pdf.add_player_info(player_data)
    
    # Check if we should split columns or just stack
    # Radar chart is the centerpiece
    pdf.add_radar_chart(radar_image_path)
    
    # Detailed stats
    pdf.add_key_stats(player_data)
    
    # AI Sentiment
    pdf.add_narrative(narrative)
    
    # Bottom metadata
    pdf.add_market_value(player_data)
    pdf.add_confidence(player_data.get('Completeness_Score', 0))
    
    # Save PDF
    try:
        pdf.output(output_path)
    except Exception as e:
        # Fallback for OS-specific file errors
        if "Permission denied" in str(e):
            import random
            alt_path = output_path.replace(".pdf", f"_{random.randint(100,999)}.pdf")
            pdf.output(alt_path)
            return alt_path
        raise e
    
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
    return generate_dossier(player_data, narrative, file_path)
