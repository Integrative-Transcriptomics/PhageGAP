/**
 * Implements an observable pattern to allow for reactive programming.
 *
 * The `Observable` class allows you to create an object that holds a value and notifies
 * subscribers when that value changes.
 */
export class Observable {
	constructor(v) {
		this.value = v;

		this.valueChangedCallback = null;

		this.setValue = function (v) {
			if (this.value != v) {
				this.value = v;
				this.raiseChangedEvent(v);
			}
		};

		this.getValue = function () {
			return this.value;
		};

		this.onChange = function (callback) {
			this.valueChangedCallback = callback;
		};

		this.raiseChangedEvent = function (v) {
			if (this.valueChangedCallback) {
				this.valueChangedCallback(v);
			}
		};
	}
}
