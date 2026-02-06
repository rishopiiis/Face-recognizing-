# generate_report.py - Generate attendance reports
import csv
import pandas as pd
from datetime import datetime

def generate_attendance_report():
    """Generate formatted attendance reports"""
    input_file = "attendance_records.csv"
    output_file = f"attendance_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
    
    if not os.path.exists(input_file):
        print("❌ No attendance file found")
        return
    
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Create pivot table
    pivot = df.pivot_table(
        index='Name',
        columns='Date',
        values='Time',
        aggfunc='count',
        fill_value='Absent'
    )
    
    # Calculate totals
    pivot['Total Present'] = pivot.apply(
        lambda row: sum([1 for x in row if x != 'Absent']), axis=1
    )
    
    # Save to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Raw Data', index=False)
        pivot.to_excel(writer, sheet_name='Summary')
        
        # Create monthly summary
        df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
        monthly = df.groupby(['Name', 'Month']).size().unstack(fill_value=0)
        monthly.to_excel(writer, sheet_name='Monthly Summary')
    
    print(f"✅ Report generated: {output_file}")
    print(f"📊 Total records: {len(df)}")
    print(f"👥 Unique people: {df['Name'].nunique()}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")

if __name__ == "__main__":
    generate_attendance_report()