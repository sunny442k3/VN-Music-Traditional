import { useEffect, useRef, useState } from 'react';
import MusicSheet from '../components/MusicSheet.component';
import Navbar from "../components/Navbar.component";
import axios from 'axios';

export default function Index() {
    // var a = ['X32 | X32 | X32 |', 'X32 | X32 | X32 |', 'X32 | X32 | X32 |', 'X32 | X32 | X32 |', 'X32 | X32 | X32|', 'X32 | X32 | X32|', 'X32 | X32 | X32|']
    const [notes, setNotes] = useState('X32\nX32\nX32\nX32\nX32\nX32');

    const [media, setMedia] = useState("2/4");
    const [title, setTitle] = useState("Title");
    
    const randint = (lim, mod) => {
        var num = Math.floor(Math.random()*(lim+1))%mod;
        return num;
    }

    const changeNumerator = (num) => {
        var new_media = media.split("/");
        new_media[0] = num.target.value.toString();
        console.log(new_media)
        if(new_media[0].length == 0){
            new_media[0] = "2";
        }
        new_media = new_media.join("/");
        setMedia(new_media);
    }
    const changeDenominator = (num) => {
        var new_media = media.split("/");
        new_media[1] = num.target.value.toString();
        if(new_media[1].length == 0){
            new_media[1] = "4";
        }
        new_media = new_media.join("/");
        setMedia(new_media);
    }
    const changeTitle = (title) => {
        var new_title = title.target.value;
        setTitle(new_title);
    }
    const predictNote = async(e) => {
        axios.get('http://localhost:5000/').then(response => {
            console.log(response);
            setNotes(response.data.text)
        });
    }

    const randomNote = (e) => {
        const NOTES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"];
        var track_length = notes.length;
        var idx_list = [];
        for(var i = 0; i < 4; i++){
            var track_idx = randint(101, 7);
            while(idx_list.includes(track_idx)){
                track_idx = randint(101, 7);
            }
            idx_list.push(track_idx);
        }
        var new_notes = ['X32 | X32 | X32 |', 'X32 | X32 | X32 |', 'X32 | X32 | X32 |', 'X32 | X32 | X32 |', 'X32 | X32 | X32|', 'X32 | X32 | X32|', 'X32 | X32 | X32|'];
        for(var idx of idx_list){
            var note_1 = randint(101, 7);
            var idx_note_1 = randint(101, 3);
            var note_2 = randint(101, 7);
            var idx_note_2 = randint(101, 3);
            var sub_track = ["X32", "X32", "X32"]
            sub_track[idx_note_1] = NOTES[note_1] + " X16";
            sub_track[idx_note_2] = NOTES[note_2] + " X16";
            var track = sub_track[0] + " | " + sub_track[1] + " | " + sub_track[2] + " |";
            new_notes[idx] = track;
        }
        setNotes(new_notes);
    }

    return (
        <div className="container-page">
            <Navbar />
            <div className="main" >
                <div className="row" style={{ height: "100%" }}>
                    <div className="col-lg-7">
                        <div style={{ height: "100%" }}>
                            <div id="sheet">
                                <div id="_sheet">
                                    <MusicSheet
                                        title={title}
                                        length='1/8'
                                        media={media}
                                        key='C trebel'
                                        notes={notes || 'abc'}
                                    ></MusicSheet>
                                </div>
                            </div>
                            <div id="_control">
                                <div id="button-control">
                                    <button><i className="fa-solid fa-backward-step"></i></button>
                                    <button><i className="fa-solid fa-play"></i></button>
                                    <button><i className="fa-solid fa-forward-step"></i></button>
                                </div>
                                <div id="time-bar">
                                    <span id="_time_start">00:00</span>
                                    <span id="_progress_bar">
                                        <i className="fa-solid fa-circle"></i>
                                    </span>
                                    <span id="_time_end">00:00</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="col-lg-5">
                        <div id="tool-bar">
                            <div id="title_sheet">
                                <div className="p-2 bd-highlight">
                                    <span>Tiêu đề</span>
                                    <input value={title} onChange={changeTitle} />
                                </div>
                            </div>

                            <div id="media">
                                <div className="d-flex justify-content-between bd-highlight">
                                    <div className="p-2 bd-highlight" id="numerator">
                                        <span>Số phách</span>
                                        <input onChange={changeNumerator} value={media.split("/")[0]} />
                                    </div>
                                    <div className="p-2 bd-highlight" id="denominator">
                                        <span>Trường độ</span>
                                        <input onChange={changeDenominator} value={media.split("/")[1]} />
                                    </div>
                                </div>
                            </div>
                            
                            <div id="make_sheet">
                                <div className="d-flex justify-content-between bd-highlight">
                                    {/* <div className="p-2 bd-highlight" id="random_note">
                                        <button onClick={randomNote}><i className="fa-solid fa-music-note"></i> Random Note</button>
                                    </div> */}
                                    <div className="p-2 bd-highlight" id="predict_note">
                                        <button onClick={predictNote}><i className="fa-regular fa-microchip-ai"></i> Fill Note</button>
                                    </div>
                                </div>
                            </div>
                            <div id="download">
                                <p>Tải file:</p>
                                <div className="d-flex justify-content-between bd-highlight">
                                    <div className="p-2 bd-highlight" id="pdf">
                                        <button><i className="fa-regular fa-file-pdf"></i> .PDF</button>
                                    </div>
                                    <div className="p-2 bd-highlight" id="midi">
                                        <button><i className="fa-regular fa-file-music"></i> .MIDI</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
