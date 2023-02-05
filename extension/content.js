const API_KEY = '';

function clearDocument() {
    document.body.innerHTML = '';
}

function showWarning() {
    document.body.innerHTML = "<h1 style=\'color:red;\'>WARNING!</h1><p>This site has been cleared due to security concerns.</p>";
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function checkURL(jobID) {
    try {
        const response = await fetch('https://developers.checkphish.ai/api/neo/scan/status', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                apiKey: API_KEY,
                jobID: jobID,
                insights: true
            })
        });
        const data = await response.json();
        console.log(data);
        return data;
    } catch (error) {
        console.error(error);
    }
}

async function checkPhishing(jobID) {
    const status = await checkURL(jobID);
    if (status.disposition !== 'clean') {
        clearDocument();
        showWarning();
    }
}


async function submitJob() {
    const currentUrl = window.location.href;
    console.log(currentUrl);
    try {
        const response = await fetch('https://developers.checkphish.ai/api/neo/scan', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                apiKey: API_KEY,
                urlInfo: {
                    url: currentUrl
                }
            })
        });
        const data = await response.json();
        console.log(data);
        // wait 3 ms
        await delay(3000);
        await checkPhishing(data.jobID);

    } catch (error) {
        console.error(error);
    }
}


submitJob();
