

// chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
//     // skip over settings and other browser tabs that aren't a website.
//     if (typeof changeInfo?.url !== 'undefined') {
//         if (!changeInfo.url.includes("chrome://")) {
//             console.log(changeInfo.url);
//         }
//     }

// }) 