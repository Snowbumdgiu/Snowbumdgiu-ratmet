import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# **SHEET 1**

import pandas as pd

# Load the sheets
sheet1_url = "https://docs.google.com/spreadsheets/d/1v31rgP0WB4FN4dC-Gr6fqZjzJRvHdBRb/export?format=csv&gid=2065875717"
sheet2_url = "https://docs.google.com/spreadsheets/d/1OPr58A3vBp5alcajzVG1UcH43rzmAiZH/export?format=csv&gid=169077083"

# Load the data into DataFrames
sheet1_df = pd.read_csv(sheet1_url)
sheet2_df = pd.read_csv(sheet2_url)

# Extract the symbol part before the '(' in sheet1
sheet1_df['Symbol'] = sheet1_df['Code'].str.split('(').str[0]

# Map the sectors from sheet2 to the symbols in sheet1
merged_df = pd.merge(sheet1_df, sheet2_df[['Symbol', 'Sector']], on='Symbol', how='left')
merged_df= merged_df.drop(columns=['Symbol'])
merged_df.head(10)



# Chuyển từ wide format sang long format
df_melted = merged_df.melt(id_vars=['Name', 'Code', 'CURRENCY','Sector'],
                    var_name='Year', value_name='Value')

# Chuyển đổi năm về kiểu số nguyên
df_melted['Year'] = pd.to_numeric(df_melted['Year'])

# Giữ nguyên thứ tự công ty gốc, tạo cột chỉ số thứ tự dựa trên thứ tự ban đầu của 'Name'
df_melted['Company'] = df_melted['Name'].factorize()[0]

# Sắp xếp dữ liệu theo thứ tự công ty
df_sorted = df_melted.sort_values(by=['Company', 'Year'])
df_sorted=df_melted.drop(columns=['Company'])
df_sorted

import numpy as np

df_sorted['Sector'] = df_sorted['Sector'].replace(['-', 'Unclassified'], [np.nan, 'Other'])
# Các dữ liệu null ở cột sector sẽ được phân loại thành other
df_sorted['Sector'] = df_sorted['Sector'].fillna('Other')
df_sorted


df_sorted.to_csv("df_sorted.csv", index=False)

#Biểu đồ 1
import pandas as pd
import plotly.express as px

# Hàm tải dữ liệu
def load_data():
    # Đọc dữ liệu từ tệp CSV
    data = pd.read_csv("df_sorted.csv")

    # Xóa khoảng trắng trong tên cột
    data.columns = data.columns.str.strip()

    # Chuẩn hóa giá trị số thập phân (dấu phẩy -> dấu chấm)
    data["Value"] = data["Value"].replace(",", ".", regex=True).astype(float)

    return data

# Load dữ liệu
data = load_data()

# Lọc dữ liệu có 'MARKET VALUE' trong cột 'Name'
market_value_data = data[data["Name"].str.contains("MARKET VALUE", case=False, na=False)]

if not market_value_data.empty:
    # Nhập năm bắt đầu và kết thúc
    min_year, max_year = market_value_data["Year"].min(), market_value_data["Year"].max()
    print(f"Data available from {min_year} to {max_year}.")

    start_year = int(input(f"Enter start year (between {min_year} and {max_year}): "))
    end_year = int(input(f"Enter end year (between {start_year} and {max_year}): "))

    if start_year >= min_year and end_year <= max_year and start_year <= end_year:
        # Lọc dữ liệu theo khoảng năm
        filtered_data = market_value_data[
            (market_value_data["Year"] >= start_year) & (market_value_data["Year"] <= end_year)
        ]

        # Vẽ biểu đồ động với plotly
        fig = px.line(
            filtered_data,
            x="Year",
            y="Value",
            color="Name",
            title="Market Value Over Time",
            labels={"Value": "Market Value", "Year": "Year", "Name": "Company"},
            hover_name="Name",  # Hiển thị tên công ty khi trỏ chuột
            hover_data={"Year": True, "Value": ":.2f"}  # Định dạng giá trị
        )

        fig.update_traces(mode="lines+markers")  # Thêm điểm vào đường
        fig.update_layout(
            hovermode="x unified",  # Thông tin hover hiển thị theo trục X
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )

        # Hiển thị biểu đồ
        fig.show()
    else:
        print(f"Invalid year range. Please choose years between {min_year} and {max_year}.")
else:
    print("No data found with 'MARKET VALUE' in the 'Name' column.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(df_sorted)

# 2. Chuyển đổi định dạng giá trị số (nếu cần)
df["Value"] = df["Value"].replace(",", ".", regex=True).astype(float)

# 3. Lọc dữ liệu liên quan đến MARKET VALUE
df_market_value = df[df["Name"].str.contains("MARKET VALUE", na=False)]

# 4. Tính tổng vốn hóa thị trường theo `Sector` và `Year`
sector_year_data = df_market_value.groupby(["Sector", "Year"])["Value"].sum().reset_index()

# 5. Tính tổng vốn hóa thị trường theo từng ngành (để lấy top 5 ngành)
sector_total = sector_year_data.groupby("Sector")["Value"].sum().reset_index()
top_5_sectors = sector_total.nlargest(5, "Value")["Sector"]

# 6. Lọc chỉ các ngành thuộc top 5
sector_year_data = sector_year_data[sector_year_data["Sector"].isin(top_5_sectors)]

# 7. Tính tỷ lệ vốn hóa từng ngành so với thị trường
sector_year_data["Total_Market_Cap"] = sector_year_data.groupby("Year")["Value"].transform("sum")
sector_year_data["Market_Cap_Percent"] = (
    sector_year_data["Value"] / sector_year_data["Total_Market_Cap"] * 100
)

# 8. Pivot dữ liệu để phù hợp với định dạng bubble chart
pivot_data = sector_year_data.pivot(index="Year", columns="Sector", values="Market_Cap_Percent").fillna(0)

# 10. Lọc ra chỉ những năm chẵn (2000, 2002, 2004,...)
selected_years = [year for year in pivot_data.index if year % 2 == 0]

# 11. Vẽ bubble chart
plt.figure(figsize=(16, 10))  # Tăng kích thước biểu đồ

# 12. Tạo bubble chart với trục tung là năm, trục hoành là ngành
colors = [
    "#FF6F61",  # Đỏ nhẹ
    "#6B8E23",  # Xanh lá cây nhẹ
    "#4682B4",  # Xanh dương nhạt
    "#FFD700",  # Vàng cam
    "#8A2BE2"   # Tím nhạt
]

# Mã màu sẽ thay đổi từ sáng đến tối theo năm
color_map = plt.cm.viridis  # Sử dụng bảng màu gradient

sector_color_map = dict(zip(pivot_data.columns, colors))  # Lưu màu cho mỗi ngành

for i, year in enumerate(selected_years):
    for j, sector in enumerate(pivot_data.columns):
        size = pivot_data.loc[year, sector]
        if size > 0:
            # Điều chỉnh màu sắc theo năm
            color = color_map(i / len(selected_years))  # Tạo màu gradient cho mỗi năm

            # Điều chỉnh kích thước bong bóng
            size_scaled = size * 50  # Kích thước bong bóng
            alpha_scaled = min(size / 50, 1)  # Độ đậm nhạt dựa vào kích thước

            plt.scatter(
                j,  # Vị trí trục x (Ngành)
                i,  # Vị trí trục y (Năm)
                s=size_scaled,  # Kích thước bong bóng
                color=color,  # Màu sắc theo năm
                alpha=alpha_scaled,  # Độ đậm nhạt dựa vào kích thước
                edgecolors="black"
            )
            # Hiển thị số liệu chỉ khi bong bóng lớn hơn ngưỡng
            if size > 15:
                plt.text(
                    j,
                    i,
                    f"{size:.1f}",
                    color="black",
                    fontsize=10,
                    ha="center",
                    va="center"
                )

# 13. Tùy chỉnh trục và nhãn
plt.xticks(range(len(pivot_data.columns)), pivot_data.columns, rotation=0, ha='center')  # Xoay cột tên ngành cho dễ đọc
plt.yticks(range(len(selected_years)), selected_years)
plt.xlabel("Sector", fontsize=14)
plt.ylabel("Year", fontsize=14)
plt.title("Top 5 Sectors by Market Value (2000-2022)", fontsize=16)
plt.grid(False)

# 14. Đảm bảo tên ngành xuống dòng nếu quá dài
for label in plt.gca().get_xticklabels():
    label.set_fontsize(10)
    label.set_wrap(True)  # Đảm bảo xuống dòng nếu cần

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_chart(file_path, company_name):
    try:


        # Hiển thị danh sách các cột trong dữ liệu
        print("Danh sách các cột trong dữ liệu:")
        print(df.columns.tolist())

        # Kiểm tra xem các cột cần thiết có tồn tại không
        required_columns = ['Name', 'Year', 'Value']  # Cần sửa tên cột theo dữ liệu thực tế
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Cột '{col}' không tồn tại trong dữ liệu.")

        # Lọc dữ liệu theo tên công ty
        company_data = df[df['Name'].str.contains(company_name, case=False)]

        if company_data.empty:
            raise ValueError(f"Không tìm thấy dữ liệu cho công ty '{company_name}'.")

        # Lọc dữ liệu cho Book Value per Share và Debt to Equity Ratio
        bvps_data = company_data[company_data['Name'].str.contains('BOOK VALUE PER SHARE', case=False)]
        debt_to_equity_data = company_data[company_data['Name'].str.contains('TOTAL DEBT % COMMON EQUITY', case=False)]

        if bvps_data.empty or debt_to_equity_data.empty:
            raise ValueError("Không tìm thấy dữ liệu Book Value per Share hoặc Debt to Equity Ratio cho công ty.")

        # Sắp xếp dữ liệu theo năm
        bvps_data = bvps_data.sort_values('Year')
        debt_to_equity_data = debt_to_equity_data.sort_values('Year')

        # Vẽ biểu đồ kết hợp
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Biểu đồ cột cho Book Value per Share
        ax1.bar(bvps_data['Year'], bvps_data['Value'], color='darkcyan', label='Book Value per Share')
        ax1.set_xlabel('Năm', fontsize=12)
        ax1.set_ylabel('Book Value per Share', fontsize=12, color='darkcyan')
        ax1.tick_params(axis='y', labelcolor='darkcyan')

        # Biểu đồ đường cho Debt to Equity Ratio
        ax2 = ax1.twinx()
        ax2.plot(debt_to_equity_data['Year'], debt_to_equity_data['Value'], color='goldenrod', marker='o', label='Debt to Equity Ratio')
        ax2.set_ylabel('Debt to Equity Ratio', fontsize=12, color='goldenrod')
        ax2.tick_params(axis='y', labelcolor='goldenrod')

        # Tiêu đề và chú thích
        plt.title(f'Mối Quan Hệ Giữa Giá Trị Sổ Sách và Tỷ Lệ Nợ trên Vốn {company_name}', fontsize=14)
        fig.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Sử dụng hàm
file_path = '/mnt/data/df_sorted.csv'  # Đường dẫn đến tệp CSV
company_name = input("Nhập tên công ty cần vẽ biểu đồ: ")
plot_combined_chart(file_path, company_name)

import pandas as pd
import matplotlib.pyplot as plt

def plot_roe_chart(file_path, country1, country2):
    try:

        # Hiển thị danh sách các cột trong dữ liệu
        print("Danh sách các cột trong dữ liệu:")
        print(df.columns.tolist())

        # Kiểm tra xem các cột cần thiết có tồn tại không
        required_columns = ['Name', 'Year', 'Value']  # Cần sửa tên cột theo dữ liệu thực tế
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Cột '{col}' không tồn tại trong dữ liệu.")

        # Lọc dữ liệu cho ROE của từng nước
        country1_data = df[(df['Name'].str.contains('RETURN ON EQUITY', case=False)) & (df['Name'].str.contains(country1, case=False))]
        country2_data = df[(df['Name'].str.contains('RETURN ON EQUITY', case=False)) & (df['Name'].str.contains(country2, case=False))]

        if country1_data.empty:
            raise ValueError(f"Không tìm thấy dữ liệu ROE cho quốc gia '{country1}'.")
        if country2_data.empty:
            raise ValueError(f"Không tìm thấy dữ liệu ROE cho quốc gia '{country2}'.")

        # Sắp xếp dữ liệu theo năm
        country1_data = country1_data.sort_values('Year')
        country2_data = country2_data.sort_values('Year')

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 6))
        plt.plot(country1_data['Year'], country1_data['Value'], marker='o', label=f'ROE - {country1}')
        plt.plot(country2_data['Year'], country2_data['Value'], marker='o', label=f'ROE - {country2}')

        plt.title(f'Return on Equity: A Comparison Between {country1} and {country2}', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('ROE (%)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Sử dụng hàm
file_path = '/mnt/data/df_sorted.csv'  # Đường dẫn đến tệp CSV
country1 = input("Nhập tên công ty đầu tiên: ")
country2 = input("Nhập tên công ty thứ hai: ")
plot_roe_chart(file_path, country1, country2)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ Google Sheets dưới dạng CSV
url = 'https://docs.google.com/spreadsheets/d/1TGIQ0CJfs3pyz7aztJDWU1KpiNw5IhuL/export?format=csv'
df = pd.read_csv(url)

# Chuyển đổi dữ liệu sang dạng dài (long format)
df = pd.melt(df, id_vars=['Name', 'Sector', 'Sector_Encoded'],
                var_name='Date', value_name='Value')

# Chuyển cột 'Date' thành kiểu datetime
df['Date'] = pd.to_datetime(df['Date'])

# Giữ nguyên thứ tự công ty gốc, tạo cột chỉ số thứ tự dựa trên thứ tự ban đầu của 'Name'
df['Company_Order'] = df['Name'].factorize()[0]

# Sắp xếp dữ liệu theo thứ tự công ty
df_sorted = df.sort_values(by=['Company_Order', 'Date'])

# Đổi thứ tự cột: Đặt 'Date' trước 'Sector', và để 2 cột 'Sector' ở cuối
df = df_sorted[['Name', 'Date', 'Value', 'Sector', 'Sector_Encoded']]

# Hiển thị dữ liệu sau khi thay đổi thứ tự cột
df.head(30)

df.to_csv("df.csv", index=False)