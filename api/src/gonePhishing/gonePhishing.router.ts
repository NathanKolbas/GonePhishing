import express, { Request, Response } from "express";
import axios from "axios";
import {Item} from "../items/item.interface";
import * as ItemService from "../items/items.service";
import {itemsRouter} from "../items/items.router";

/**
 * Router Definition
 */
export const gonePhishingRouter = express.Router();

// GET items/:url
gonePhishingRouter.get("/", async (req: Request, res: Response) => {
    const url: string = req.body.url
    console.log(url);

    try {
        const response = await axios.post(`http://127.0.0.1:5000/${url}`);
        
        if (response) {
            return res.status(200).send(response);
        }

        res.status(404).send("item not found");
    } catch (e) {
        // @ts-ignore
        res.status(500).send(e.message);
    }
});

gonePhishingRouter.post("/", async (req: Request, res: Response) => {
    const url: string = req.body.url
    console.log(url);

    try {
        const response = await axios.post(`http://127.0.0.1:5000/${url}`);

        if (response) {
            console.log(response.data);
            return res.status(200).send(JSON.stringify(response.data));
        }

        res.status(404).send("item not found");
    } catch (e) {
        // @ts-ignore
        res.status(500).send(e.message);
    }
});
