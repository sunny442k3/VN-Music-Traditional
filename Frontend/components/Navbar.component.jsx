import { useRef, useEffect } from 'react';
import Link from 'next/link'

const Navbar = () => {

	return <div className="header">
            <link
                rel="stylesheet"
                href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css"
                integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T"
                crossOrigin="anonymous"
            />
            <link
                rel="stylesheet"
                href="https://site-assets.fontawesome.com/releases/v6.0.0/css/all.css" async
            />
            <div className="row no-gutters">
                <div style={{width:"100%"}}>
                    <div className="d-flex justify-content-between bd-highlight mb-3">
                        <div className="p-2 bd-highlight" id="logo-page">
                            <i className="fa-solid fa-list-music"></i><Link href="/"> F.Muse</Link>
                        </div>
                        <div className="p-2 bd-highlight" id="about-page">
                            <Link href="">Liên hệ</Link>
                            <Link href="./wiki">Tài liệu</Link>
                        </div>
                    </div>
                </div>
            </div>
        </div>
};

export default Navbar;
