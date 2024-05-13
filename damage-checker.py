import pandas as pd

def check_damage(csv_file, output_file):
    data = pd.read_csv(csv_file)
    rows_with_damage_above_70 = []

    for i in range(len(data)):
        if data.loc[i, 'PREDICTED_DAMAGE'] > 70:
            consecutive_count = 1
            for j in range(i+1, min(i+11, len(data))):
                if data.loc[j, 'PREDICTED_DAMAGE'] > 70:
                    consecutive_count += 1
                else:
                    break
            if consecutive_count == 10:
                rows_with_damage_above_70.append(data.loc[i:i+9])

    if len(rows_with_damage_above_70) > 0:
        result_df = pd.concat(rows_with_damage_above_70)
        print("Rows with consecutive damage above 70:")
        print(result_df)
        result_df.to_csv(output_file, index=False)
        print(f"Saved result to {output_file}")
    else:
        print("No consecutive rows with damage above 70 found.")

# Load data
check_damage('P3-300-prediction.csv', 'P3-300-checked.csv')