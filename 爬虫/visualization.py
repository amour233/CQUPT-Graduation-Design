from wordcloud import WordCloud

from utils import setup_plt

plt = setup_plt()


def show_ip_location(df):
    """
    IP属地可视化
    :param df: 表
    :return:
    """
    data = df["IP归属地"].value_counts()
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.values, marker='o')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def show_senti_distribution(df):
    """
    情感倾向分布可视化
    :param df: 表
    :return:
    """
    data = df["情感倾向"].value_counts()
    plt.figure(figsize=(10, 5))
    plt.plot(data.values, data.index, marker='o')
    plt.title("情感倾向分布")
    # x轴只显示2位小数
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.axhline(0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()


def show_region_senti(df):
    """
    地区情感倾向可视化
    :param df: 表
    :return:
    """
    data = df.groupby("IP归属地")["平均情感倾向"].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.values, marker='o')
    plt.title("地区平均情感倾向")
    # x轴只显示2位小数
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.xticks(rotation=90)
    plt.axhline(0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.figure(figsize=(10, 6))
    plt.show()

def show_gender_senti(df):
    """
    性别情感倾向可视化
    :param df: 表
    :return:
    """
    data = df.groupby("性别")["平均情感倾向"].mean()
    plt.bar(data.index, data.values)
    plt.title("性别平均情感倾向")
    plt.xlabel('性别')
    plt.ylabel('值')
    plt.show()

def show_svip_senti(df):
    """
    svip情感倾向可视化
    :param df: 表
    :return:
    """
    data = df.groupby("svip")["平均情感倾向"].mean()
    plt.bar(data.index, data.values)
    tick_locations = [i + 0.5 for i in range(len(data.index))]
    # 刻度标签：与categories相同
    plt.xticks(tick_locations, data.index)
    # 设置x轴的刻度间隔为1
    plt.xticks(range(len(data.index)), data.index)
    plt.title("svip平均情感倾向")
    plt.xlabel('svip')
    plt.ylabel('值')
    plt.show()

def show_senti_support(positive, negative):
    """
    正向情感和负向情感的点赞可视化
    :param negative: 负向情感
    :param positive: 正向情感
    :return:
    """
    plt.pie([positive, negative], labels=["正向情感支持率", "负向情感支持率"], autopct='%1.1f%%')
    plt.title("正向情感和负向情感的支持率")
    plt.tight_layout()
    plt.show()


def show_wordcloud(words):
    """
    生成词云
    :param words: 词表
    :return:
    """
    wordcloud = (
        WordCloud(font_path="PingFang Regular.ttf", background_color="white", max_words=2000, height=1080, width=1920)
        .fit_words(words))
    plt.figure(figsize=(19.2, 10.8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
