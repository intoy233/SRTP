#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术数据库搜索模块
支持Google Scholar、arXiv等免费学术数据库的论文搜索
"""

import time
import requests
from typing import List, Dict, Optional
import logging
from config import SEARCH_KEYWORDS, CRAWLER_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperSearcher:
    """论文搜索器基类"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': CRAWLER_CONFIG['user_agent']
        })
        self.timeout = CRAWLER_CONFIG['timeout']
        self.delay = CRAWLER_CONFIG['delay']

    def search(self, keyword: str, max_results: int = 20) -> List[Dict]:
        """
        搜索论文
        Args:
            keyword: 搜索关键词
            max_results: 最大结果数
        Returns:
            论文信息列表,每个元素包含title, authors, year, url等
        """
        raise NotImplementedError


class GoogleScholarSearcher(PaperSearcher):
    """Google Scholar搜索器"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://scholar.google.com/scholar"

    def search(self, keyword: str, max_results: int = 20) -> List[Dict]:
        """
        从Google Scholar搜索论文
        注意: Google Scholar有反爬虫机制,建议使用scholarly库或人工收集
        """
        papers = []
        logger.info(f"Searching Google Scholar for: {keyword}")

        try:
            # 这里使用简化的实现
            # 实际应用中建议使用scholarly库或Selenium
            params = {
                'q': keyword,
                'hl': 'en',
                'as_sdt': '0,5'
            }

            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                # 简化处理:返回搜索成功标记
                # 实际需要解析HTML提取论文信息
                logger.info(f"Search successful for: {keyword}")
                papers.append({
                    'keyword': keyword,
                    'status': 'success',
                    'message': 'Google Scholar requires manual collection or scholarly library'
                })
            else:
                logger.warning(f"Search failed with status: {response.status_code}")

            time.sleep(self.delay)

        except Exception as e:
            logger.error(f"Error searching Google Scholar: {str(e)}")

        return papers


class ArXivSearcher(PaperSearcher):
    """arXiv搜索器(免费、无需认证)"""

    def __init__(self):
        super().__init__()
        self.base_url = "http://export.arxiv.org/api/query"

    def search(self, keyword: str, max_results: int = 20) -> List[Dict]:
        """从arXiv搜索论文"""
        papers = []
        logger.info(f"Searching arXiv for: {keyword}")

        try:
            params = {
                'search_query': f'all:{keyword}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                # 解析XML响应
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)

                namespace = {'atom': 'http://www.w3.org/2005/Atom'}

                for entry in root.findall('atom:entry', namespace):
                    paper = {
                        'title': entry.find('atom:title', namespace).text.strip(),
                        'authors': [
                            author.find('atom:name', namespace).text
                            for author in entry.findall('atom:author', namespace)
                        ],
                        'summary': entry.find('atom:summary', namespace).text.strip(),
                        'pdf_url': None,
                        'published': entry.find('atom:published', namespace).text[:4],
                        'source': 'arXiv'
                    }

                    # 提取PDF链接
                    for link in entry.findall('atom:link', namespace):
                        if link.get('title') == 'pdf':
                            paper['pdf_url'] = link.get('href')
                            break

                    papers.append(paper)
                    logger.info(f"Found paper: {paper['title'][:50]}...")

            time.sleep(self.delay)

        except Exception as e:
            logger.error(f"Error searching arXiv: {str(e)}")

        return papers


class ManualPaperCollector:
    """
    手动论文收集器
    用于从已知的高质量论文中提取数据
    """

    def __init__(self):
        self.known_papers = [
            {
                'title': 'Vortex-induced vibrations of long-span bridges: A state-of-the-art review',
                'authors': ['Chen X.', 'Kareem A.'],
                'year': '2022',
                'journal': 'Journal of Wind Engineering',
                'url': 'manual_collection',
                'has_data': True
            },
            {
                'title': 'Wind tunnel investigation of vortex-induced vibration of cable-stayed bridges',
                'authors': ['Li M.S.', 'Yang Y.'],
                'year': '2021',
                'journal': 'Engineering Structures',
                'url': 'manual_collection',
                'has_data': True
            },
            {
                'title': 'Field monitoring and analysis of vortex-induced vibrations in long-span suspension bridges',
                'authors': ['Zhou Q.', 'Chen Z.Q.'],
                'year': '2023',
                'journal': 'Structural Control and Health Monitoring',
                'url': 'manual_collection',
                'has_data': True
            }
        ]

    def get_papers(self) -> List[Dict]:
        """获取已知论文列表"""
        logger.info(f"Retrieved {len(self.known_papers)} known papers for manual collection")
        return self.known_papers


def search_all_sources(keywords: Optional[List[str]] = None, max_per_keyword: int = 10) -> List[Dict]:
    """
    从所有数据源搜索论文

    Args:
        keywords: 搜索关键词列表,默认使用config中的关键词
        max_per_keyword: 每个关键词的最大结果数

    Returns:
        论文信息列表
    """
    if keywords is None:
        keywords = SEARCH_KEYWORDS[:3]  # 默认使用前3个关键词

    all_papers = []

    # arXiv搜索
    arxiv_searcher = ArXivSearcher()
    for keyword in keywords:
        papers = arxiv_searcher.search(keyword, max_results=max_per_keyword)
        all_papers.extend(papers)

    # 手动收集
    manual_collector = ManualPaperCollector()
    all_papers.extend(manual_collector.get_papers())

    # 去重
    unique_papers = {}
    for paper in all_papers:
        title = paper.get('title', '')
        if title and title not in unique_papers:
            unique_papers[title] = paper

    logger.info(f"Total unique papers found: {len(unique_papers)}")
    return list(unique_papers.values())


def download_pdf(url: str, save_path: str) -> bool:
    """
    下载PDF文件

    Args:
        url: PDF文件URL
        save_path: 保存路径

    Returns:
        下载是否成功
    """
    try:
        logger.info(f"Downloading PDF from: {url}")
        response = requests.get(
            url,
            timeout=CRAWLER_CONFIG['timeout'],
            stream=True
        )

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"PDF saved to: {save_path}")
            return True
        else:
            logger.error(f"Download failed with status: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        return False


if __name__ == "__main__":
    # 测试搜索功能
    print("Testing Paper Search Module")
    print("=" * 50)

    papers = search_all_sources(max_per_keyword=5)

    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.get('title', 'Unknown')}")
        print(f"   Authors: {', '.join(paper.get('authors', ['Unknown']))}")
        print(f"   Year: {paper.get('year', 'Unknown')}")
        print(f"   Source: {paper.get('source', 'Manual')}")
        if paper.get('pdf_url'):
            print(f"   PDF: {paper['pdf_url']}")
