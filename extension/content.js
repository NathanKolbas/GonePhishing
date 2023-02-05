const API_KEY = '';

function clearDocument() {
    document.body.innerHTML = '';
}

function showWarning() {
    document.body.innerHTML = "<h1 style=\'color:red;\'>WARNING!</h1><p>This site has been cleared due to security concerns.</p>";
}


async function checkIfPhishing() {
    const currentUrl = window.location.href;
    console.log(currentUrl);
    try {
        const response = await fetch(`localhost:7000/api/gonePhishing/${currentUrl}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                urlInfo: {
                    url: currentUrl
                }
            })
        });
        const data = await response.json();
        console.log(data);

        if (data.result !== 'pass') {
            clearDocument();
            showWarning();
        }

    } catch (error) {
        console.error(error);
    }
}


checkIfPhishing();
