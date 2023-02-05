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

function linkChecker() {
    // Inject the styles
    document.body.innerHTML = `<style>
.tooltip {
  // position: relative;
  // display: inline-block;
  // border-bottom: 1px dotted black;
}

.tooltiptext {
  z-index: 99999;
}

.tooltip .tooltiptext {
  visibility: hidden;
  width: 220px;
  background-color: rgb(222,222,222);
  color: #000000;
  text-align: center;
  border-radius: 6px;
  padding: 6px;
  
  /*top: 100%;*/
  /*left: 50%;*/
  margin-left: -110px; /* Use half of the width (120/2 = 60), to center the tooltip */
  
  opacity: 0;
  transition: opacity 200ms;
  
  box-shadow: none;

  /* Position the tooltip */
  position: absolute;
  
  display: flex;
  justify-content: center;
}

.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
  box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}

.tooltip .tooltiptext::after {
  content: " ";
  position: absolute;
  bottom: 100%;  /* At the top of the tooltip */
  left: 50%;
  margin-left: -5px;
  border-width: 5px;
  border-style: solid;
  border-color: transparent transparent rgb(222,222,222) transparent;
}

.loader {
  border: 7px solid #f3f3f3;
  border-top: 7px solid #3498db;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 2s linear infinite;
  text-align: center;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.bad {
    color: red;
}

.good {
    color: green;
}
</style>
${document.body.innerHTML}`;
    const links = document.body.getElementsByTagName("a");
    const loadedAttr = 'loading-data';
    for (let i = 0; i < links.length; i++) {
        const linkElement = links.item(i);
        const href = linkElement.href;

        // console.log(href.length);
        // if (href.length === 0) return;

        // Add the tooltip class to the link
        linkElement.className = `${linkElement.className} tooltip`
        // Add the tooltip element
        linkElement.innerHTML = `${linkElement.innerHTML}
<div class="tooltiptext">
    <div class="loader"></div>
</div>
`;
        // <p>${href}</p>
        linkElement.addEventListener("pointerenter", function() {
            const element = linkElement.firstElementChild;
            const alreadyLoaded = element.getAttribute(loadedAttr);
            console.log(alreadyLoaded);
            if (alreadyLoaded) return;

            element.setAttribute(loadedAttr, 'true');
            checkIfPhishing(href).then(response => {
                console.log(response);
                const phishing = response['phishing'];
                const confidence = response['confidence'];
                const text = phishing ? `Warning: ${(confidence * 100).toFixed(2)}% likely to be a phishing website` :
                    `Checked with a ${(confidence * 100).toFixed(2)}% chance of being a phishing site`;
                element.innerHTML = `<p style="margin: 0;font-size: small;" class="${phishing ? 'bad' : ''}">${text}</p>`;
            });
        });
    }
}

linkChecker();
