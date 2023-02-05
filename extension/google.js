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

async function processElement(element) {
    const href = element.href;
    const response = await checkIfPhishing(href);
    console.log(response);
    const phishing = response['phishing'];
    const confidence = response['confidence'];
    const text = phishing ? `Warning: ${(confidence * 100).toFixed(2)}% likely to be a phishing website` :
        `Checked with a ${(confidence * 100).toFixed(2)}% chance of being a phishing site`;

    // Add the tooltip element
    element.innerHTML = `${element.innerHTML} <div class="google-phishing">
    <div class="dot ${phishing ? 'bad' : 'good'}"></div>
    <p class="${phishing ? 'bad-text' : ''}">${text}</p>
</span>`;
}

async function run() {
    // Inject the styles
    document.body.innerHTML = `<style>
.google-phishing {
    display: flex;
    align-items: center;
    gap: 8px;
    color: wheat;
}

.google-phishing:hover {
    text-decoration: none;
}

.dot {
    height: 10px;
    width: 10px;
    border-radius: 50%;
}

.bad {
    background-color: red;
}

.good {
    background-color: green;
}

.bad-text {
    color: red;
}
</style>
${document.body.innerHTML}`;
    const a = document.body.getElementsByTagName("a");
    console.log(a);
    for (let i = 0; i < a.length; i++) {
        const element = a.item(i);
        if (element.innerHTML.match('<h3') === null) continue;

        processElement(element);
    }
}

run();
