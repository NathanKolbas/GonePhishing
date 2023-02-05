import axios from "axios";
import { API_KEY } from "..";

/**
 * Service Methods
 */
 
const delay = (ms: number): Promise<void> => {
    return new Promise(resolve => setTimeout(resolve, ms));
};


async function checkJob(jobID: string): Promise<any> {
    try {
        const response = await axios.post('https://developers.checkphish.ai/api/neo/scan/status', {
            apiKey: API_KEY,
            jobID: jobID,
            insights: true
        });
        return response.data;
    } catch (error) {
        console.error(error);
    }
}


async function checkPhishing(jobID: string): Promise<string> {
    const status = await checkJob(jobID);
    return status.disposition;
}

async function submitJob(url: string): Promise<string> {
    try {
        const response = await axios.post('https://developers.checkphish.ai/api/neo/scan', {
            apiKey: API_KEY,
            urlInfo: {
                url: url
            }
        });
        const data = response.data;
        // console.log(data);
        // wait 3 ms
        await delay(1000);
        const result = await checkPhishing(data.jobID);
        return result;

    } catch (error) {
        console.error(error);
    }
    return '';
}


export const checkUrl = async (url: string): Promise<string> => {
    return submitJob(url);
};