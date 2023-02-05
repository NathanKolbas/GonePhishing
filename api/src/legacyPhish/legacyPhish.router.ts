import express, { Request, Response } from "express";
import * as phishService from "./legacyPhish.service";

/**
 * Router Definition
 */
export const legacyPhishRouter = express.Router();

/**
 * Controller Definitions
 */
// GET items
legacyPhishRouter.get("/:url", async (req: Request, res: Response) => {
    const url: string = req.params.url;
    try {
        const result: string = await phishService.checkUrl(url);

        res.status(200).send(result);
    } catch (e) {
        // @ts-ignore
        res.status(500).send(e.message);
    }
});