from weibo import schedule_tasks

if __name__ == "__main__":
    search_keywords = ["唐人街探案"]
    schedule_tasks(search_keywords, interval=600)
    input("按任意键退出")
