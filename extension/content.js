const API_KEY = '';

function clearDocument() {
    document.body.innerHTML = '';
}

function showWarning(confidence=null) {
    const continueAnywayButtonId = 'continueanyway';
    const originalContent = document.body.innerHTML;
    const text = confidence == null ? `<p style="font-size: x-large;">This site has been blocked due to security concerns.</p>` :
        `<p style="font-size: x-large;">We are ${(confidence * 100).toFixed(2)}% confident that this website might be phishing. Proceed with caution.</p>`;

    clearDocument();
    document.body.innerHTML = `<style>
.continue-anyway:hover {
  cursor: pointer;
}
</style>
<div style="
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
">
    <h1 style="color:red;font-size: xxx-large;font-weight: bold;">WARNING!</h1>
    ${text}
    <a class="continue-anyway" style="color:#1843c1;" id="${continueAnywayButtonId}">Not recommended: continue anyway</a>
</div>`;
    document.getElementById(continueAnywayButtonId).addEventListener("click", function() {
        console.log(originalContent);
        document.body.innerHTML = originalContent;
    });
}

async function checkIfPhishing(url) {
    console.log(url);
    try {
        const response = await fetch(`http://127.0.0.1:7000/api/phishing/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                url: url
            })
        });
        const data = await response.json();
        console.log(data);

        return data;
    } catch (error) {
        console.error(error);
    }
}

async function checkForPhishing() {
    const currentUrl = window.location.href;
    console.log(currentUrl);
    try {
        const response = await checkIfPhishing(currentUrl);

        if (response['phishing']) {
            showWarning(response['confidence']);
        }
    } catch (error) {
        // console.error(error);
    }
}


checkForPhishing();
