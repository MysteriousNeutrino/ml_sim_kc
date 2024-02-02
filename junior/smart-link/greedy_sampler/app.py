import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()
# как идея создать ещё переменную хранящую общее кол-во записей
# делать это в классе и иметь setter и getter
# 1. click_id : [offer_id]
# 2. offer_id : count(click_id)
# 3. offer_id : count(conversion)
# 4. offer_id : sum(reward)

click_count = 0

# offer_id: list[click_id:int]
offer_clicks = {}

# click_id:int : reward:int
click_reward = {}

# offer_id : sum(reward)
offer_total_reward = {}

@app.on_event("startup")
async def startup_event():
    global click_count, offer_clicks, click_reward, offer_total_reward

    click_count = 0
    # offer_id: list[click_id:int]
    offer_clicks = {}
    # click_id:int : reward:int
    click_reward = {}
    # offer_id : sum(reward)
    offer_total_reward = {}


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """Get feedback for particular click"""
    # Response body consists of click ID
    # and accepted click status (True/False)
    if click_id not in click_reward:
        try:
            offer_id = [offer_id for offer_id, click_ids in offer_clicks.items()
                        if click_id in click_ids][0]
            offer_id = int(offer_id)

            is_conversion = reward != 0
            click_reward[click_id] = reward
            if offer_id not in offer_total_reward:
                offer_total_reward[offer_id] = 0
            offer_total_reward[offer_id] += reward
            # print("offer_total_reward: ", offer_total_reward)
        except IndexError as e:
            # В случае, если offer_id пустой (нет совпадений)
            raise HTTPException(status_code=404, detail="Offer not found") from e
    else:
        raise HTTPException(status_code=404, detail="This click already has a record of reward")

    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": is_conversion,
        "reward": reward
    }
    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""
    if offer_id not in offer_clicks:
        response = {
            "offer_id": offer_id,
            "clicks": 0,
            "conversions": 0,
            "reward": 0,
            "cr": 0,
            "rpc": 0,
        }
    else:
        clicks = offer_clicks[offer_id]
        rewarding_clicks = [click_reward[click_id] for click_id in clicks
                            if click_id in click_reward]
        conversions = len(rewarding_clicks) - rewarding_clicks.count(0)
        reward = sum(rewarding_clicks)
        print("rewarding_clicks: ", rewarding_clicks)
        print("click_reward: ", click_reward)
        print("click_count: ", click_count)
        response = {
            "offer_id": offer_id,
            "clicks": len(clicks),
            "conversions": conversions,
            "reward": reward,
            "cr": conversions / len(clicks) if conversions is not None else 0,
            "rpc": reward / len(clicks) if reward is not None else 0,
        }

    return response


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Greedy sampling"""
    # Parse offer IDs
    offer_ids = [int(offer) for offer in offer_ids.split(",")]

    # Sample top offer ID
    if click_count < 100:
        offer_id = int(np.random.choice(offer_ids))
        sampler = "random"
    else:
        # реализовать алгоритм максимизации RPC
        # брать прост offer с самым большим RPC в 90% случаев
        sampler = "greedy"
        if random.randint(0, 1) > 0.9:
            common_keys = set(offer_ids) & set(offer_total_reward.keys())
            offer_id = offer_ids[0]
            max_value = offer_total_reward[offer_id] \
                if offer_id in offer_total_reward else float('-inf')
            for key in common_keys:
                if key in offer_total_reward and offer_total_reward[key] > max_value:
                    max_value = offer_total_reward[key]
                    offer_id = key

        else:
            offer_id = int(np.random.choice(offer_ids))
    # print("offer_ids: ", offer_ids)
    add_click_to_offer(click_id, offer_id)

    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "sampler": sampler,
    }

    return response


def add_click_to_offer(click_id, offer_id):
    """
    add_click_to_offer
    :param click_id:
    :param offer_id:
    :return:
    """
    global click_count
    global offer_clicks
    if check_number_in_values(offer_clicks, click_id):
        raise HTTPException(status_code=404, detail="This click_id is already registered")
    if offer_id not in offer_clicks:
        offer_clicks[offer_id] = []
    offer_clicks[offer_id].append(click_id)
    click_count += 1
    print(offer_clicks)
    print(click_reward)


def check_number_in_values(dictionary, number):
    """
    check_number_in_values
    :param dictionary:
    :param number:
    :return:
    """
    for _, value in dictionary.items():
        if number in value:
            return True
    return False


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", reload=True)


if __name__ == "__main__":
    main()
