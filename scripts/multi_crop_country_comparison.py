import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
# Countries selected for comparison based on your africa_multicrop_overlap.csv
COUNTRIES = ["Zambia", "Tanzania", "Ethiopia", "Nigeria"]
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

def generate_multi_country_comparison():
    """
    Simulates and visualizes the 'Risk Underestimation' if only Maize 1 is tracked
    versus the 'Total Food System Envelope' across 4 key countries.
    """
    
    # Data synthesized from your uploaded africa_multicrop_overlap.csv
    # Focus on the 'Middle' or 'Southwest' regions where overlap is highest
    data = {
        "Country": ["Zambia", "Tanzania", "Ethiopia", "Nigeria"],
        "Focus_Region": ["Central", "Southwest", "Tigray", "Middle"],
        "Maize_Window_Days": [152, 212, 230, 92],  # Length of Maize 1 calendar
        "Envelope_Window_Days": [152, 212, 245, 229], # Length of All-Crops calendar
        "Extra_Crops": [0, 4, 3, 3], # Number of overlapping crops
        "Crop_List": ["Maize only", "Millet, Rice, Wheat, Sorghum", "Sorghum, Teff, Wheat", "Rice 1, Rice 2, Sorghum"]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Window Expansion (The "Risk Gap")
    df['Expansion_Days'] = df['Envelope_Window_Days'] - df['Maize_Window_Days']
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.3)

    # --- PANEL 1: CROP DIVERSITY GAP ---
    ax1 = axes[0]
    bars = ax1.bar(df['Country'], df['Extra_Crops'], color=COLORS, alpha=0.8, edgecolor='black')
    ax1.set_title("Food System Complexity:\nExtra Staple Crops Sharing the Maize Window", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Number of Additional Staple Crops")
    ax1.grid(axis='y', alpha=0.3)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'+{int(height)} crops', ha='center', fontweight='bold')

    # --- PANEL 2: CALENDAR RISK GAP ---
    ax2 = axes[1]
    x = np.arange(len(df['Country']))
    width = 0.35
    
    ax2.bar(x - width/2, df['Maize_Window_Days'], width, label='Maize 1 Window', color='silver', edgecolor='black')
    ax2.bar(x + width/2, df['Envelope_Window_Days'], width, label='Total Food Envelope', color=COLORS, alpha=0.7, edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Country'])
    ax2.set_ylabel("Length of Growing Season (Days)")
    ax2.set_title("The 'Risk Gap':\nMaize-only vs. Total Agricultural Envelope", fontsize=12, fontweight='bold')
    ax2.legend()
    
    # Highlight Nigeria's massive gap
    ax2.annotate('Tracking Maize misses\n137 days of risk to\nRice & Sorghum', 
                 xy=(3.15, 150), xytext=(2.2, 280),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    plt.suptitle("Dissertation Phase 4: Why Multi-Crop Envelopes are Essential for Food Security Analysis", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig("multi_country_crop_overlap_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Generate Comparative Report ---
    print("="*60)
    print("COMPARATIVE RISK ENVELOPE ANALYSIS")
    print("="*60)
    for _, row in df.iterrows():
        print(f"COUNTRY: {row['Country']} ({row['Focus_Region']} Region)")
        print(f"  - Maize 1 Status: {row['Maize_Window_Days']} days exposure.")
        print(f"  - Food System Status: +{row['Extra_Crops']} overlapping staples ({row['Crop_List']})")
        if row['Expansion_Days'] > 0:
            print(f"  - CRITICAL FINDING: Tracking only Maize misses {row['Expansion_Days']} days of drought risk.")
        else:
            print(f"  - FINDING: Maize is a 100% reliable proxy for total system risk.")
        print("-" * 60)

if __name__ == "__main__":
    generate_multi_country_comparison()