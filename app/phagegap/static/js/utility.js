/**
 * Displays a notification toast with a custom message to the user.
 *
 * @param {string} message The message to display.
 * @param {string} title The title of the notification.
 * @param {string} style The style of the notification ("info", "success", "warning", "alert").
 * @param {boolean} keepOpen Whether the notification should stay open until manually closed.
 */
export function displayNotification(
	message,
	title = "Notification",
	style = "alert",
	keepOpen = true,
) {
	let icon = "";
	switch (style) {
		case "info":
			icon = "fa-solid fa-circle-info";
			break;
		case "success":
			icon = "fa-solid fa-circle-check";
			break;
		case "warning":
			icon = "fa-solid fa-triangle-exclamation";
			break;
		case "alert":
			icon = "fa-solid fa-circle-info";
			break;
		default:
			break;
	}
	title = `<i class="${icon}"></i> ${title}`;
	message += `<br><span class="text-small text-center reduce-3">Click to close.</span>`;
	style += " mono";
	Metro.notify.create(message, title, {
		width: 400,
		keepOpen: keepOpen,
		clsNotify: style,
		animation: "easeIn",
	});
}

/**
 * Legacy conform function that creates a file input element and clicks it to prompt the user to select a file.
 *
 * This function returns a Promise that resolves with the selected file in a FormData object.
 * If the user cancels the file selection, the Promise is rejected.
 *
 * @param {string} accept - The file type(s) to accept (e.g., ".gff", ".gff,.gff3").
 * @returns {Promise<FormData>} A Promise that resolves with the selected file in a FormData object.
 */
export function promptFile(accept) {
	return new Promise((resolve, reject) => {
		const fileInput = document.createElement("input");
		fileInput.type = "file";
		fileInput.accept = accept;
		fileInput.onchange = () => {
			if (fileInput.files.length > 0) {
				const formData = new FormData();
				formData.append("file", fileInput.files[0]);
				resolve(formData);
			} else {
				reject(new Error("No file selected."));
			}
			fileInput.remove(); // Clean up the input element after use.
		};
		fileInput.click(); // Trigger the file selection dialog.
	});
}

/**
 * Downloads a Blob object as a file with the specified name.
 *
 * @param {Blob} blob The data to be downloaded as a Blob.
 * @param {string} name The name of the file to download.
 */
export function downloadBlob(blob, name) {
	var download_link = document.createElement("a");
	download_link.href = window.URL.createObjectURL(blob, { type: "text/plain" });
	download_link.download = name;
	download_link.click();
	download_link.remove();
}

/**
 * Reads a client side file to a {@link String}.
 *
 * @param {string} file Client side local file path.
 * @returns {Promise<string>} File content as a {@link String}.
 */
export function readFile(file) {
	return new Promise((resolve, reject) => {
		var fileReader = new FileReader();
		fileReader.onload = (event) => {
			resolve(event.target.result);
		};
		fileReader.onerror = (error) => reject(error);
		fileReader.readAsText(file);
	});
}

/**
 * Compresses a string using the specified encoding (e.g., "gzip", "deflate", "brotli").
 * 
 * Source: https://gist.github.com/Explosion-Scratch/357c2eebd8254f8ea5548b0e6ac7a61b
 * 
 * @param {string} string The string to compress.
 * @param {string} encoding The encoding to use for compression (e.g., "gzip", "deflate", "brotli").
 * @returns {Promise<ArrayBuffer>} A Promise that resolves with the compressed data as an ArrayBuffer.
 */
export function compress(string, encoding) {
	const byteArray = new TextEncoder().encode(string);
	const cs = new CompressionStream(encoding);
	const writer = cs.writable.getWriter();
	writer.write(byteArray);
	writer.close();
	return new Response(cs.readable).arrayBuffer().then(function (arrayBuffer) {
		return arrayBuffer;
	});
}

/**
 * Decompresses a byte array using the specified encoding (e.g., "gzip", "deflate", "brotli").
 * 
 * Source: https://gist.github.com/Explosion-Scratch/357c2eebd8254f8ea5548b0e6ac7a61b
 * 
 * @param {*} byteArray The byte array to decompress. 
 * @param {string} encoding The encoding to use for decompression (e.g., "gzip", "deflate", "brotli").
 * @returns 
 */
export function decompress(byteArray, encoding) {
	const cs = new DecompressionStream(encoding);
	const writer = cs.writable.getWriter();
	writer.write(byteArray);
	writer.close();
	return new Response(cs.readable).arrayBuffer().then(function (arrayBuffer) {
		return new TextDecoder().decode(arrayBuffer);
	});
}

/**
 * Capitalizes the first letter of a given text.
 *
 * @param {string} text The text to capitalize.
 * @returns {string} The capitalized text.
 */
export function capitalize(text) {
	return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * Converts a string to title case.
 *
 * The function splits the input text into words using the specified separator
 * (defaulting to whitespace, hyphens, and underscores), capitalizes the first
 * letter of each word, and then joins them back together with spaces.
 *
 * @param {string} text The text to titleize.
 * @param {RegExp} separator The regular expression to split the text into words.
 * @returns {string} The titleized text.
 */
export function titleize(text, separator = /[\s-_]/) {
	return text
		.split(separator)
		.map((word) => capitalize(word))
		.join(" ");
}

/**
 * Converts a string to lowercase and replaces spaces, hyphens, and underscores with hyphens.
 *
 * The function splits the input text into words using the specified separator (defaulting to
 * whitespace, hyphens, and underscores), converts each word to lowercase, and then joins them
 * back together with hyphens.
 *
 * @param {string} text The text to untitleize.
 * @param {RegExp} separator The regular expression to split the text into words.
 * @returns {string} The untitleized text.
 */
export function untitleize(text, separator = /[\s-_]/) {
	return text
		.split(separator)
		.map((word) => word.toLowerCase())
		.join("-");
}

/**
 * Calculates the sum of an array of numbers.
 *
 * @param {Array<number>} arr The array of numbers to sum.
 * @returns {number} The sum of the numbers in the array.
 */
export function sum(arr) {
	let sum = 0;
	let i = -1;
	while (++i < arr.length) {
		sum += arr[i];
	}
	return sum;
}

/*
 * > Network related utility functions.
 */

/**
 * Displays a notification based on the status of a Flask response object.
 *
 * @param {object} response Flask response object.
 */
export function notifyResponse(response) {
	if (response === undefined || response.status === undefined) {
		return;
	}
	var title;
	switch (response.status) {
		case 200:
			title = "Success";
			break;
		case 400:
			title = "Bad Request";
			break;
		case 403:
			title = "Forbidden";
			break;
		case 500:
			title = "Internal Server Error";
			break;
		case 504:
			title = "Gateway Timeout";
			break;
		default:
			title = "Error";
	}
	displayNotification(
		response.data.error || response.data.message || "An error occurred while processing your request.",
		title,
		"alert mono",
	);
}

/**
 * Promise check for the presence of the `phagegap_consent` cookie.
 * 
 * Displays a cookie consent notification if the cookie is not found that resolves
 * the promise when the user accepts the cookie consent. If the cookie is already
 * present, it is refreshed with the same value and updated expiration time and the
 * promise is resolved immediately.
 */
export function cookieDisclaimer() {
	return new Promise((resolve, reject) => {
		if (!checkCookie()) {
			// No cookie is found, display the cookie disclaimer notification.
			Metro.cookieDisclaimer.init({
				name: "phagegap_consent",
				clsContainer: "phagegap-cookie-disclaimer",
				title: "This site uses cookies 🍪",
				message: `
			To use PhageGap you must accept the use of cookies.
			This website uses cookies to securely process your data.
			Your data will not be accessible to us or third parties and is not stored on our servers.
			You will not be tracked and no personal information will be collected.
			`,
				duration: "30days",
				onAccept: function () {
					Metro.cookie.setCookie("phagegap_consent", "accepted", {
						domain: window.hostname,
						secure: true,
						maxAge: 30 * 24 * 60 * 60,
						samesite: "Strict",
					});
					resolve(); // Resolve the promise when the user accepts the cookie consent.
				},
				onDecline: function () {
					reject(); // Reject the promise if the user declines.
				}
			});
		} else {
			// Refresh the cookie with the same value and updated expiration time.
			var gCookie = Metro.cookie.getCookie("phagegap_consent");
			Metro.cookie.setCookie("phagegap_consent", "accepted", {
				domain: window.hostname,
				secure: true,
				maxAge: 30 * 24 * 60 * 60,
				samesite: "Strict",
			});
			resolve(); // Resolve the promise immediately if the cookie is already present.
		}
	});
}

/**
 * Checks for the presence of a `phagegap_consent` cookie.
 *
 * @returns True if the cookie is present, false otherwise.
 */
export function checkCookie() {
	var gCookie = Metro.cookie.getCookie("phagegap_consent");
	return gCookie !== null && gCookie !== undefined;
}

/**
 * Registers event listeners for socket communication.
 *
 * This function sets up event listeners on the provided socket object to handle events emitted by
 * the server.
 *
 * @param {Object} socket The socket object to register event listeners on.
 */
export function registerSocket(socket) {
	socket.on("notify", (content) => {
		displayNotification(content.message, content.title, "bg-light", false);
	});
}
