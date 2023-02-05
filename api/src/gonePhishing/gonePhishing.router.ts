import express, { Request, Response } from "express";
import axios from "axios";

/**
 * Router Definition
 */
export const gonePhishingRouter = express.Router();

// GET items/:url
gonePhishingRouter.get("/:url", async (req: Request, res: Response) => {
    const url: string = req.params.url

    try {
        const response = await axios.post(`localhost:8080/${url}`);
        
        if (response) {
            return res.status(200).send(response);
        }

        res.status(404).send("item not found");
    } catch (e) {
        // @ts-ignore
        res.status(500).send(e.message);
    }
});