import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from geopandas import GeoDataFrame

# 假设您有一个包含IP归属地和情感标签的数据集
# 加载情感数据
kiwi_data = pd.read_csv('data/kiwi_9000s_comments_with_labels.csv')

# 确保数据包含IP归属地和情感标签列
if 'IP归属地' not in kiwi_data.columns or 'label' not in kiwi_data.columns:
    raise ValueError("Data must contain 'IP归属地' and 'label' columns")

# 加载世界地图数据
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# 创建一个函数来解析IP归属地并添加地理位置信息
def get_geolocation(location_name):
    # 使用Google Maps API来获取地理位置的经纬度信息
    # 请确保您已经注册了Google Maps API并获得了API密钥
    # 替换下面的API密钥为您自己的密钥
    google_maps_api_key = 'YOUR_GOOGLE_MAPS_API_KEY'
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location_name}&key={google_maps_api_key}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        return data['results'][0]['geometry']['location']
    else:
        return {'latitude': None, 'longitude': None}

# 应用函数到IP归属地列
kiwi_data = kiwi_data.apply(lambda x: x.fillna(get_geolocation(x['IP归属地'])) if pd.isna(x['IP归属地']) else x, axis=1)

# 创建一个情感分数的函数
def calculate_sentiment_score(group):
    # 这里是一个示例函数，根据情感标签计算分数
    # 您可以根据实际的情感模型调整这个函数
    if group['label'].iloc[0] == '积极':
        return 1
    elif group['label'].iloc[0] == '消极':
        return -1
    else:
        return 0

kiwi_data['sentiment_score'] = kiwi_data.groupby('IP归属地')['label'].apply(calculate_sentiment_score)

# 创建一个GeoDataFrame
gdf = GeoDataFrame(kiwi_data, geometry='geometry')

# 合并情感数据到世界地图
world = world.set_index('name').join(gdf.set_index('IP归属地'), how='left')

# 绘制世界地图
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax)
world['sentiment_score'].plot(kind='scatter', ax=ax, color='blue', alpha=0.3)
plt.title('世界情感地图')
plt.show()
