import { useRef, useEffect } from 'react';

const MusicSheet = ({ title, media, length, key, notes }) => {
	const ref = useRef();
	const sheet = typeof window !== undefined ? require('abcjs') : null;

	useEffect(() => {
		sheet.renderAbc(ref.current, `X: 2\nT: ${title}\nM: ${media}\nL: 4\nK: ${key}\n${notes}`);
	}, [sheet, key, title, length, notes, media]);

	return <div id="_row_sheet" ref={ref}></div>;
};

export default MusicSheet;
