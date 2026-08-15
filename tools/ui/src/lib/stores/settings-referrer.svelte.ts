import { ROUTES } from '$lib/constants';

let _url = $state<string>(ROUTES.SETTINGS_EXIT);

export const settingsReferrer = {
	get url() {
		return _url;
	},
	set url(value: string) {
		_url = value;
	}
};
