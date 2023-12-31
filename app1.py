from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import logging
import os
import pickle
import re
import string
import time

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Request
from fastapi_utils.tasks import repeat_every
from scipy.spatial import distance
import base64
from PIL import Image

# import scipy
from config import *
from image_extractor import ImageExtractor
from search import SearchIndex
from sentence_vectorizer import SentenceVectorizer
from io import BytesIO

app = FastAPI()
template = Jinja2Templates(
    directory="template"
)  # Create a templates directory for HTML files
logger = logging.getLogger(__name__)

sv = SentenceVectorizer()
ie = ImageExtractor()
search_index = SearchIndex()

templates_list = []
templates = pd.DataFrame()
popular_memes = pd.DataFrame()


# Define your HTML templates in the 'templates' directory.


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return template.TemplateResponse("home.html", {"request": request})


def check_image(filename):
    for ext in ALLOWED_IMAGE_EXTENSIONS:
        if filename.endswith(ext):
            return True
    return False


@app.get("/initialize")
@app.on_event("startup")
def initialize():
    reload_index()
    reload_templates()
    reload_sentence_vectorizer()


@app.on_event("shutdown")
def shutdown_event():
    save_index()


# @repeat_every(seconds=10) # 10 min
def save_index():
    # Save the data to preserve the meme views (popular)
    search_index.data.to_pickle(SEARCH_INDEX_FILENAME)
    logger = logging.getLogger("uvicorn.info")
    logger.info(f"Success saving index data: {SEARCH_INDEX_FILENAME}")
    return "Success saving index data"


@app.get("/reload_sentence_vectorizer")
def reload_sentence_vectorizer():
    global sv
    logger = logging.getLogger("uvicorn.info")
    try:
        # Download if the embeddings file does not exist
        if not os.path.isfile(EMBEDDINGS_FILENAME):
            logger.info(f"Downloading embeddings from {EMBEDDINGS_URL}")
            r = requests.get(url=EMBEDDINGS_URL, allow_redirects=True)
            with open(EMBEDDINGS_FILENAME, "wb") as f:
                f.write(r.content)
            logger.info(f"Embeddings saved to {EMBEDDINGS_FILENAME}")

        # Load the embeddings
        sv.load(EMBEDDINGS_FILENAME)
        logger.info(f"Success loading vectors: {EMBEDDINGS_FILENAME}")
        return "Success loading vectors"
    except:
        logger = logging.getLogger("uvicorn.error")
        logger.error("Failed to load vectors")
        return "Failed to load vectors"


@app.get("/reload_index")
def reload_index():
    global search_index
    try:
        search_index.load(filename=SEARCH_INDEX_FILENAME, reader_fn=SEARCH_READER_FN)
        search_index.build(search_cols=SEARCH_COLUMNS, max_dims=SEARCH_MAX_DIMS)
        # search_index.data['text'] = search_index.data['text'].fillna(value='')
    except:
        logger = logging.getLogger("uvicorn.error")
        logger.error("Failed to load index")
        return "Failed to load index"

    logger = logging.getLogger("uvicorn.info")
    logger.info(f"Success loading index: {SEARCH_INDEX_FILENAME}")
    return "Success loading index"


@app.get("/reload_templates")
def reload_templates():
    global templates
    try:
        templates = pd.read_pickle(TEMPLATES_INDEX_FILENAME)
        templates = templates.reset_index()
    except:
        logger = logging.getLogger("uvicorn.error")
        logger.error("Failed to load templates")

    logger = logging.getLogger("uvicorn.info")
    logger.info(f"Success loading templates: {TEMPLATES_INDEX_FILENAME}")
    return "Success loading templates"


@app.get("/autocomplete")
def autocomplete(request: Request, query_text: str, col: str = "title"):
    if len(query_text) >= 2:
        # Get matching entries for title and text content
        rows_title = search_index.data[
            search_index.data["title"].str.contains(query_text, flags=re.IGNORECASE)
        ]
        # rows_text = search_index.data[search_index.data['text'].str.contains(query_text, flags=re.IGNORECASE)]

        # Combine them and drop duplicates
        rows = rows_title
        # rows = pd.concat([rows_title, rows_text]).drop_duplicates(subset=['id'])
        results_with_images = []
        for idx, row in rows.iterrows():
            result = {
                "idx": idx,
                "name": row["title"],
                "text": row["text"],
                "url": row["url"],
                "views": row["views"],
                "website": row["website"],
            }
            img_url = row["url"]
            result["image_url"] = img_url
            results_with_images.append(result)

            results_with_images.insert(0, result)
        return template.TemplateResponse(
            "search_result.html", {"request": request, "results": results_with_images}
        )
    return {"message": "Error, query is empty!"}
    #     return [
    #         {
    #             "idx": idx,
    #             "id": row["id"],
    #             "url": row["url"],
    #             "name": row["title"],
    #             "text": row["text"],
    #             "website": row["website"],
    #         }
    #         for idx, row in rows.iterrows()
    #     ]
    # return []


@app.get("/total_memes")
def total_memes():
    return {"total_memes": len(search_index.data)}


@app.get("/popular_memes")
def get_popular_memes(request: Request, count: int = 20):
    # popular_df = search_index.data.sort_values(by='views', ascending=False)
    # popular_df = popular_df[0:count]
    popular_df = search_index.data.nlargest(count, columns=["views"])
    results_with_images = []
    for idx, row in popular_df.iterrows():
        result = {
            "idx": idx,
            "name": row["title"],
            "text": row["text"],
            "url": row["url"],
            "views": row["views"],
            "website": row["website"],
        }
        img_url = row["url"]
        result["image_url"] = img_url
        results_with_images.append(result)

        results_with_images.insert(0, result)
    return template.TemplateResponse(
        "search_result.html", {"request": request, "results": results_with_images}
    )

    # return [
    #     {
    #         "idx": idx,
    #         "id": row["id"],
    #         "url": row["url"],
    #         "name": row["title"],
    #         "text": row["text"],
    #         "views": row["views"],
    #         "website": row["website"],
    #     }
    #     for idx, row in popular_df.iterrows()
    # ]


@app.get("/templates")
def get_templates(page: int, items_per_page: int):
    global templates_list

    # Calculate only the first time. Use global cache after first request
    if len(templates_list) == 0:
        templates_list = []
        for i, row in templates.iterrows():
            templates_list.append(
                {"idx": i, "id": row["id"], "url": row["url"], "name": row["title"]}
            )

    return {
        "templates": templates_list[
            page * items_per_page : (page + 1) * items_per_page
        ],
        "total_pages": (len(templates_list) // items_per_page) + 1,
        "page": page,
        "items_per_page": items_per_page,
    }


@app.get("/search")
def search(
    request: Request,
    query: str,
    count: int = 20,
    mode: str = "title",
    threshold: float = 1.0,
    add_views: bool = False,
):
    if mode == "both":
        search_column = "fusion_text_glove"
    if mode == "title":
        search_column = "title_glove"
    if mode == "content":
        search_column = "ocr_glove"
    if mode in ["image", "url", "template"]:
        search_column = "img_embedding"

    if query:
        start = time.time()

        if mode == "url":
            if not check_image(query):
                return {"results": [], "valid_results": 0}

            r = requests.get(query)
            img_name = os.path.join(TEMP_IMAGES_DIR, query.split("/")[-1])
            with open(img_name, "wb") as f:
                f.write(r.content)
            query_embedding = ie.to_vec(filename=img_name, to_numpy=True)

        elif mode == "template":
            img_entry = templates.iloc[int(query)]
            query_embedding = img_entry[search_column]
            query_embedding = np.asarray(query_embedding)

        elif mode == "image":
            img_entry = search_index.data.iloc[int(query)]
            query_embedding = img_entry[search_column]
            query_embedding = np.asarray(query_embedding)

            # Used in the meme details page only
            if add_views:
                search_index.data["views"].iloc[int(query)] += 1
        else:
            # Calculate the embedding of the query
            query_embedding = sv.encode(query)

        # Perform the query in the index
        query_results, scores = search_index.query(
            vector=query_embedding,
            col=search_column,
            k=count,
            return_scores=True,
            # threshold=threshold # NOTE: In practice, it is better to apply the threshold after
        )
        valid_results = 0
        for score in scores:
            valid_results += int(score > 0 and score <= threshold)

        # Put results in adequate format
        # results = []
        # for (i, item), score in zip(query_results.iterrows(), scores):
        #     if score <= threshold:
        #         results.append({
        #             'idx': i,
        #             'name': item['title'],
        #             'text': item['text'],
        #             'url': item['url'],
        #             'score': score
        #         })

        # # Insert the template itself as first result in this mode
        # if mode == 'template':
        #     temp = templates.iloc[int(query)]
        #     results.insert(0, {
        #         'idx': int(query),
        #         'name': temp['title'],
        #         'text': "",
        #         'url': temp['url'],
        #         'score': 0
        #     })

        results_with_images = []
        for (i, item), score in zip(query_results.iterrows(), scores):
            if score <= threshold:
                result = {
                    "idx": i,
                    "name": item["title"],
                    "text": item["text"],
                    "score": score,
                }

                if mode in ["url", "template", "image", "title"]:
                    img_url = item["url"]
                    # result["image_url"] = img_url
                    if check_image(img_url):
                        result["image_url"] = img_url
                    else:
                        result[
                            "image_url"
                        ] = "default_image_url.jpg"  # Provide a default image URL for invalid URLs
                else:
                    result[
                        "image_url"
                    ] = "default_image_url.jpg"  # Provide a default image URL for other modes

                results_with_images.append(result)

        # Insert the template itself as the first result in this mode
        if mode == "template":
            temp = templates.iloc[int(query)]
            result = {"idx": int(query), "name": temp["title"], "text": "", "score": 0}

            if check_image(temp["url"]):
                result["image_url"] = temp["url"]
            else:
                result[
                    "image_url"
                ] = "default_image_url.jpg"  # Provide a default image URL for invalid URLs

            results_with_images.insert(0, result)

        logger = logging.getLogger("uvicorn.info")
        logger.info(
            f'QUERY: "{query}" ({np.round(time.time()-start, 4)} s., top-{count})'
        )

        return template.TemplateResponse(
            "search_result.html", {"request": request, "results": results_with_images}
        )

    return {"message": "Error, query is empty!"}


if __name__ == "__main__":
    import argparse

    import uvicorn

    # Launch app
    uvicorn.run(app, host="192.168.43.206", port=8080, log_level="info")
