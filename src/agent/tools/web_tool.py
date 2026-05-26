import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from zai import ZhipuAiClient

load_dotenv()  # 加载环境变量

@tool("web_search")
def web_search(query: str) -> str:
    """
    一个网络搜索工具函数,你可以使用它来搜索互联网上的信息。

    Args:
        query: 搜索查询字符串
        
    Returns:
        返回的搜索结果
    """
    try:
        client = ZhipuAiClient(
            api_key=os.getenv("ZHIPUAI_API_KEY")
        )
        response = client.web_search.web_search(
            search_engine="search_pro",
            search_query=query,  
            search_recency_filter="noLimit",
        )
        if response.search_result:
            return "\n\n".join([item.content for item in response.search_result])
        else:
            return "No results found."
    except Exception as e:
        return f"Error occurred while searching: {e}"
    
if __name__ == "__main__":
    res = web_search.invoke({"query": "2026年4月6日上证指数的股价是多少？"})
    print(res)