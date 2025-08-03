import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from charts.plot_ppie import plot_professional_pie
from charts.plot_pie_newclor import plot_color_pie
from charts.plot_hbar import plot_hbar_chart_
from charts.plot_bar import plot_bar_chart
from charts.plot_line import plot_line_chart
from charts.plot_scatter import plot_scatter_chart
from charts.plot_histogram import plot_histogram_chart
from charts.plot_cat_bar import plot_cbar_chart
from matplotlib.backends.backend_pdf import PdfPages
df=pd.read_csv('fleet_fuel_data.csv')
df=pd.DataFrame(df)
# print(df.info())
''' giving figure to 1st dashbord'''
figs,axis=plt.subplots(2,2,figsize=(13,9),facecolor="#d0d4d1")
plt.suptitle("General Fuel Overview & Analysis", fontsize=18, fontweight='bold',)
'''1st dashboard for general overview '''
df['L_required_per_Km']=df['Fuel_Consumed_Litres']/df['Distance_Travelled_KM']
df['Fule_efficiency']=df['Distance_Travelled_KM']/df['Fuel_Consumed_Litres']
# fuel types & concumption in liters
fuel_types=(df.groupby(['Fuel_Type'])['Fuel_Consumed_Litres'].sum().sort_values(ascending=False)).round(1)
print(fuel_types)
plot_color_pie(fuel_types, title='Fuel Type & its Usage', top_n=5, value_label='Liters',
                          label_font='Verdana', title_font='Arial', show_others=False, start_angle=140,
                          ax=axis[0,1],pie_radius=1.3, pie_center=(0.8, -0.5))
#  verage fuel consumption by vehicle type (e.g., truck, van).
avg_fuel_by_vechile_type=(df.groupby(['Asset_Type'])['Fuel_Consumed_Litres'].mean().sort_values(ascending=False)).round(1)
# print(avg_fuel_by_vechile_type)

plot_cbar_chart(
    avg_fuel_by_vechile_type.index,
    avg_fuel_by_vechile_type.values,
    title=' Avg Fuel Consumption by Asset Type',
    title_font='Arial',
    xlabel='Asset Type',
    ylabel='Avg Consumption(L)',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,             
    use_sns_palette=True,             
    sns_palette="colorblind",               
    rotation=0,
    show_minor_ticks=True,
    show_minor_labels=False,
    ax=axis[0,0]
)


df['Trip_Date']=pd.to_datetime(df['Trip_Date'])
df['Month']=df['Trip_Date'].dt.to_period('M')
monthly_trend = df.groupby('Month')['Fuel_Consumed_Litres'].sum().reset_index()
monthly_trend['Month'] = monthly_trend['Month'].dt.to_timestamp()
monthly_trend=monthly_trend.sort_values('Month')
# print(monthly_trend)

plot_line_chart(
    monthly_trend['Month'],
    monthly_trend['Fuel_Consumed_Litres'],
    title='Monthly Trend In Fuel Consumption',
    xlabel='Month',
    ylabel='Consumption(L)',
    label='Montly_Consumption(L)',
    linestyle='solid',
    linewidth=2,
    color="#1f77b4",
    marker='o',
    markersize=6,
    markerfacecolor='white',
    markeredgecolor='black',
    legend=True,
    ax=axis[1,1],
    rotation=25,
    annotation_freq=3,
    show_rolling_avg=True,
    rolling_window=3,
    rolling_color='green'
)

# plt.show()
#  thsi is for weekly trend in fuel consumption
df['Week']=df['Trip_Date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_trend=df.groupby(['Week'])['Fuel_Consumed_Litres'].sum().reset_index()
weekly_trend=weekly_trend.sort_values('Week')
plot_line_chart(
    weekly_trend['Week'],
    weekly_trend['Fuel_Consumed_Litres'],
    title='Weekly Trend In Fuel Consumption',
    xlabel='Week',
    ylabel='Consumption(L)',
    label='Weekly_Consumption(L)',
    linestyle='solid',
    linewidth=2,
    color="#8f561c",
    marker='o',
    markersize=5,
    markerfacecolor='white',
    markeredgecolor='black',
    legend=True,
    ax=axis[1,0],
    rotation=25,
    annotation_freq=5,
)
plt.subplots_adjust(
    left=0.07,    
    right=0.96,   
    top=0.87,     
    bottom=0.1,   
    wspace=0.25,   
    hspace=0.45 
)
# plt.show()

''' this is dashboard no 2 or weather specifically '''

fig,axas=plt.subplots(2,2,figsize=(13,9),facecolor="#d0d4d1")
plt.suptitle("Weather Impact on Fuel & Trip Efficiency", fontsize=18, fontweight='bold')
avg_consumptionvsweather=(df.groupby(['Weather_Condition'])['Fuel_Consumed_Litres'].mean().sort_values(ascending=False)).round(1)
# print(avg_consumptionvsweather)
plot_cbar_chart(
    avg_consumptionvsweather.index,
    avg_consumptionvsweather.values,
    title='Avg Fuel Consumption in Different Weathers',
    title_font='Arial',
    xlabel='Weather Condition',
    ylabel='Fuel Consumption(L)',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="colorblind",               
    rotation=0,
    show_minor_ticks=True,
    show_minor_labels=False,
    ax=axas[0,0]
)

# i dont know why but grid lines were not working for this plot so i just do it manuallly by using th below codes
axas[0,0].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axas[0,0].grid(False, which='major', axis='x')
axas[0,0].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')
box = axas[0,0].get_position()  
axas[0,0].set_position([box.x0 + 0.05, box.y0, box.width, box.height]) 
# plt.show()

# trip status vs weather type 
tripstatusvswweather=df.groupby(['Weather_Condition','Trip_Status']).size().unstack().fillna(0)
# print(tripstatusvswweather)
sns.set_palette("bright"),
tripstatusvswweather.plot(
    kind='bar',
    ax=axas[0,1],
    figsize=(10,6),
    edgecolor='black',
)
axas[0,1].minorticks_on()
axas[0,1].facecolor='#f5f5f5'
axas[0,1].set_title('Trip Status by Weather Condition', fontsize=12, fontweight='semibold',family='Arial')
axas[0,1].set_xlabel('Weather Condition',fontsize=11,family='Verdana')
axas[0,1].set_ylabel('Number of Trips',fontsize=11,family='Verdana')
axas[0,1].legend(title='Trip Status',fontsize=6,title_fontsize=7,loc='lower right')
axas[0,1].tick_params(axis='x', rotation=0)
axas[0,1].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axas[0,1].grid(False, which='major', axis='x')
axas[0,1].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')

offset = tripstatusvswweather.max().max() * 0.015
# Annotate each bar (loop over container)
for container in axas[0,1].containers:
    for bar in container:
        height = bar.get_height()
        axas[0,1].text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=6,
            fontweight='normal',
            family='Tahoma',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )
df['Distance_Travelled_KM']=(df['Distance_Travelled_KM']).round(1)
''' how much d/f types of fuel is used per km in different weather conditions'''
eff_in_weather=(df.groupby(['Weather_Condition','Fuel_Type'])['L_required_per_Km'].mean().unstack()).round(2)
# print(eff_in_weather)
eff_in_weather.plot(
    colormap="viridis",
    kind='bar',
    ax=axas[1,0],
    figsize=(10,6),
    edgecolor='black',
)
axas[1,0].minorticks_on()
axas[1,0].facecolor='#f5f5f5'
axas[1,0].set_title('Avg(L/km) by Fuel Type in different Weather Conditions', fontsize=11, fontweight='semibold',family='Arial')
axas[1,0].set_xlabel('Weather Condition',fontsize=11,family='Verdana')
axas[1,0].set_ylabel('Liters Per km',fontsize=11,family='Verdana')
axas[1,0].legend(title='Fule_Type',fontsize=6,title_fontsize=7,loc='lower right')
axas[1,0].tick_params(axis='x', rotation=0)
axas[1,0].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axas[1,0].grid(False, which='major', axis='x')
axas[1,0].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')



offset = eff_in_weather.max().max() * 0.015
# Anotate each bar (loop over container)
for container in axas[1,0].containers:
    for bar in container:
        height = bar.get_height()
        axas[1,0].text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f'{float(height)}',
            ha='center',
            va='bottom',
            fontsize=6,
            fontweight='normal',
            family='Tahoma',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )

# plt.show()

'''Weather condition vs avg trip duration in hours per km '''
df = df[df["Distance_Travelled_KM"] != 0]
df['time_per_1km']=((df['Trip_Duration_Hours']/df['Distance_Travelled_KM'])*60)
Avg_time_m_per_km=df.groupby(['Asset_Type','Weather_Condition'])["time_per_1km"].mean().reset_index()
Avg_time_m_per_km['time_per_1km']=Avg_time_m_per_km['time_per_1km'].round(2)
# print(Avg_time_m_per_km)

heatmap_data=Avg_time_m_per_km.pivot(
    index='Asset_Type',
    columns='Weather_Condition',
    values='time_per_1km'
)
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.2f',
    cmap='YlGnBu',
    linewidths=0.5,
    linecolor='gray',
    ax=axas[1,1],
    cbar_kws={'label':'Avg Minutes per KM'}
)
axas[1, 1].set_title("Avg Time (min) to Travel 1 KM by Vehicle Type and Weather", fontsize=12, fontweight='semibold', pad=10,family='Arial')
axas[1, 1].set_xlabel("Weather Condition", fontsize=11, fontweight='normal',family='Verdana')
axas[1, 1].set_ylabel("Asset Type", fontsize=11, fontweight='normal',family='Verdana')
axas[1, 1].tick_params(axis='x', rotation=0)
axas[1, 1].tick_params(axis='y', rotation=0)
plt.subplots_adjust(
    left=0.075,    
    right=0.98,   
    top=0.88,     
    bottom=0.08,  
    wspace=0.25,   
    hspace=0.45 
)
# Moving the bar plot at axas[1,0] to the left
box = axas[1,0].get_position()  
axas[1,0].set_position([box.x0 - 0.02, box.y0, box.width, box.height]) 

'''Dashboard no 3 '''
fig3,axs=plt.subplots(2,2,figsize=(13,9),facecolor="#d0d4d1")
plt.suptitle("Driver Details & Trip Efficiency Dashboard", fontsize=18, fontweight='bold')
Effeciency_By_driver=(df.groupby(['Driver_ID'])['Fule_efficiency'].mean().sort_values(ascending=False).head(10)).round(2)
print(Effeciency_By_driver)
plot_hbar_chart_(
    Effeciency_By_driver.index,
    Effeciency_By_driver.values,
    title='Top 10 Most Fuel Efficient Drivers',
    title_font='Arial',
    xlabel='Fuel Efficiency(km/L)',
    ylabel='Driver ID',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="colorblind",
    show_minor_ticks=True,
    show_minor_labels=False,
    annotate=True,
    grid=True,
    ax=axs[1,0]
)
axs[1,0].minorticks_on()
axs[1,0].tick_params(axis='x', rotation=0)
axs[1,0].grid(True, which='major', axis='x', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axs[1,0].grid(False, which='major', axis='y')
axs[1,0].grid(True, which='minor', axis='x',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')


'''AVg trip cost per km by different drivers '''
df['cost_per_km']=(df['Total_Trip_Cost']/df['Distance_Travelled_KM'])
driver_perkm_cost=(df.groupby(['Driver_ID'])['cost_per_km'].mean().sort_values(ascending=False).head(10)).round(2)

plot_hbar_chart_(
    driver_perkm_cost.index,
    driver_perkm_cost.values,
    title='Top 10 Most Expensive Drivers',
    title_font='Arial',
    xlabel='Cost (AED/km)',
    ylabel='Driver ID',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="colorblind",
    show_minor_ticks=True,
    show_minor_labels=False,
    annotate=True,
    grid=True,
    ax=axs[0,0]
)
axs[0,0].minorticks_on()
axs[0,0].tick_params(axis='x', rotation=0)
axs[0,0].grid(True, which='major', axis='x', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axs[0,0].grid(False, which='major', axis='y')
axs[0,0].grid(True, which='minor', axis='x',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')



'''least expensive drivers '''
driver_perkm_lcost=(df.groupby(['Driver_ID'])['cost_per_km'].mean().sort_values(ascending=False).tail(10)).round(2)

plot_hbar_chart_(
    driver_perkm_lcost.index,
    driver_perkm_lcost.values,
    title='Top 10 Least Expensive Drivers',
    title_font='Arial',
    xlabel='Cost (AED/km)',
    ylabel='Driver ID',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="colorblind",
    show_minor_ticks=True,
    show_minor_labels=False,
    annotate=True,
    grid=True,
    ax=axs[0,1]
)
axs[0,1].minorticks_on()
axs[0,1].tick_params(axis='x', rotation=0)
axs[0,1].grid(True, which='major', axis='x', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axs[0,1].grid(False, which='major', axis='y')
axs[0,1].grid(True, which='minor', axis='x',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')

'''avg cost of the efficeint drivers per km'''
top_drivers=Effeciency_By_driver.index
top_eff_cost=df[df['Driver_ID'].isin(top_drivers)].groupby('Driver_ID')['cost_per_km'].mean().round(2)
top_eff_cost=top_eff_cost.sort_values(ascending=True)
plot_hbar_chart_(
    top_eff_cost.index,
    top_eff_cost.values,
    title='Cost(AED/km) of Top 10 Most Fuel Efficient Drivers',
    title_font='Arial',
    xlabel='Cost (AED/km)',
    ylabel='Driver ID',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="colorblind",
    show_minor_ticks=True,
    show_minor_labels=False,
    annotate=True,
    grid=True,
    ax=axs[1,1]
)
axs[1,1].minorticks_on()
axs[1,1].tick_params(axis='x', rotation=0)
axs[1,1].grid(True, which='major', axis='x', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axs[1,1].grid(False, which='major', axis='y')
axs[1,1].grid(True, which='minor', axis='x',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')
plt.subplots_adjust(
    left=0.08,    
    right=0.98,   
    top=0.87,     
    bottom=0.09,   
    wspace=0.23,   
    hspace=0.45 
)

'''Dashboard no 4 for asset age and maninatance vs cost '''
fig4,axes=plt.subplots(2,2,figsize=(13,9),facecolor="#d0d4d1")
plt.suptitle("Asset Age, Maintenance & Cost Impact", fontsize=18, fontweight='bold')
Aage_vs_fuel_consumption=(df.groupby(['Asset_Age_Years'])['Fuel_Consumed_Litres'].mean().sort_values(ascending=False)).round(2)
sns.regplot(
    x=Aage_vs_fuel_consumption.index,
    y=Aage_vs_fuel_consumption.values,
    ax=axes[1,0],
    scatter_kws={'s':60, 'color':'#D55E00'},
    line_kws={'color':'#0072B2', 'linewidth':2},
    ci=95
)
axes[1,0].set_title('Asset Age vs Avg Fuel Consumption(L)', fontsize=11, fontweight='semibold',family='Arial')
axes[1,0].set_xlabel('Asset Age (Years)')
axes[1,0].set_ylabel('Average Fuel Consumption (L)')
axes[1,0].grid(True, linestyle="--", alpha=0.6)
axes[1,0].minorticks_on()
axes[1,0].grid(True, which='minor', axis='y', linestyle=":", linewidth=0.4, zorder=0)


# Annotate each scatter point with its y-value (fuel consumption)
for x, y in zip(Aage_vs_fuel_consumption.index, Aage_vs_fuel_consumption.values):
    axes[1,0].annotate(
        f"{y:.1f}",              
        xy=(x, y),               
        xytext=(0, 5),           
        textcoords="offset points",
        ha='center',
        fontsize=6,
        color='black',
        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2') 
    )
box = axes[1,0].get_position()  
axes[1,0].set_position([box.x0 + 0.05, box.y0, box.width, box.height]) 

''' Asset type and its speed for maintained and unmaintained conditions '''
df['Speed_KMPH'] = pd.to_numeric(df['Speed_KMPH'], errors='coerce')
avg_speed=df.groupby(['Asset_Type','Maintenance_Flag'])['Speed_KMPH'].mean().unstack()
sns.set_palette("colorblind"),
avg_speed.plot(
    kind='bar',
    ax=axes[0,0],
    figsize=(10,6),
    edgecolor='black',
)
axes[0,0].minorticks_on()
axes[0,0].facecolor='#f5f5f5'
axes[0,0].set_title('Avg Speed of different Assets & Maintanace', fontsize=12, fontweight='semibold',family='Arial')
axes[0,0].set_xlabel('Asset Type',fontsize=11,family='Verdana')
axes[0,0].set_ylabel('Speed(km/hr)',fontsize=11,family='Verdana')
axes[0,0].legend(title='Maintenance_Flag',fontsize=6,title_fontsize=7,loc='lower right')
axes[0,0].tick_params(axis='x', rotation=0)
axes[0,0].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axes[0,0].grid(False, which='major', axis='x')
axes[0,0].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')

offset = avg_speed.max().max() * 0.015
# Anotte each bar (loop over container)
for container in axes[0,0].containers:
    for bar in container:
        height = bar.get_height()
        axes[0,0].text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=6,
            fontweight='normal',
            family='Tahoma',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )
''' avg fuel consumption vs maintannce in different asset'''
avg_fuel_maintance=df.groupby(['Asset_Type','Maintenance_Flag'])['Fuel_Consumed_Litres'].mean().unstack()
sns.set_palette("colorblind"),
avg_fuel_maintance.plot(
    kind='bar',
    ax=axes[0,1],
    figsize=(10,6),
    edgecolor='black',
)
axes[0,1].minorticks_on()
axes[0,1].facecolor='#f5f5f5'
axes[0,1].set_title('Avg Fuel Consumption of different Assets & Maintanace', fontsize=11, fontweight='semibold',family='Arial')
axes[0,1].set_xlabel('Asset Type',fontsize=11,family='Verdana')
axes[0,1].set_ylabel('Avg Fuel COnsumption(L)',fontsize=11,family='Verdana')
axes[0,1].legend(title='Maintenance_Flag',fontsize=6,title_fontsize=7,loc='lower right')
axes[0,1].tick_params(axis='x', rotation=0)
axes[0,1].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axes[0,1].grid(False, which='major', axis='x')
axes[0,1].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')

offset = avg_fuel_maintance.max().max() * 0.015
# annotate each bar (loop over container)
for container in axes[0,1].containers:
    for bar in container:
        height = bar.get_height()
        axes[0,1].text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=6,
            fontweight='normal',
            family='Tahoma',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )
'''Asset typen and sum of trip cost by antance flag'''
Asset_Type_Trip_cost=df.groupby(['Asset_Type','Maintenance_Flag'])['cost_per_km'].mean().unstack()
sns.set_palette("colorblind"),
avg_fuel_maintance.plot(
    kind='bar',
    ax=axes[1,1],
    figsize=(10,6),
    edgecolor='black',
)
axes[1,1].minorticks_on()
axes[1,1].facecolor='#f5f5f5'
axes[1,1].set_title('Avg Cost by Asset Type & Maintanace', fontsize=11, fontweight='semibold',family='Arial')
axes[1,1].set_xlabel('Asset Type',fontsize=11,family='Verdana')
axes[1,1].set_ylabel('Total Cost(AED)',fontsize=11,family='Verdana')
axes[1,1].legend(title='Maintenance_Flag',fontsize=6,title_fontsize=7,loc='lower right')
axes[1,1].tick_params(axis='x', rotation=0)
axes[1,1].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axes[1,1].grid(False, which='major', axis='x')
axes[1,1].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')

offset = Asset_Type_Trip_cost.max().max() * 0.002
for container in axes[1,1].containers:
    for bar in container:
        height = bar.get_height()
        axes[1,1].text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=6,
            fontweight='normal',
            family='Tahoma',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )
plt.subplots_adjust(
    left=0.075,    
    right=0.98,   
    top=0.88,     
    bottom=0.08,  
    wspace=0.25,   
    hspace=0.45 
) 
# with PdfPages('Dashboard_Fleet_Feul_Analysis.pdf') as pdf:
#     pdf.savefig(figs)
#     pdf.savefig(fig)
#     pdf.savefig(fig3)
#     pdf.savefig(fig4)
plt.show()