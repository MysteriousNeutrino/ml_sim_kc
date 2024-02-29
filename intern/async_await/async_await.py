import asyncio
import random

import httpx
from fastapi import FastAPI

app = FastAPI()


@app.post("/parse_url/")
async def parse_url(url: str) -> str:
    """
    parse_url
    :param url:
    :return:
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            await r.raise_for_status()

            parse_time = 0.1 * random.randint(5, 10) if random.random() < 0.1 else 0.1
            await asyncio.sleep(parse_time)

            return f"Parsed {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


async def run_test(n_requests: int) -> float:
    """
    run_test
    :param n_requests:
    :return:
    """
    base_url = "https://httpbin.org/"
    # перенести baseurl в httpx.AsyncClient
    async with httpx.AsyncClient(app=app) as client:
        ts = asyncio.get_event_loop().time()
        await asyncio.gather(
            *[client.post(url=f"{base_url}/parse_url/") for _ in range(n_requests)])
        return asyncio.get_event_loop().time() - ts


if __name__ == "__main__":
    t = asyncio.run(run_test(n_requests=10000))
    print(f"Time taken: {t} seconds")
