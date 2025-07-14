import { useState } from "react";
import { increment, decrement , incrementByAmount } from "./features/counterSlice";
import { useDispatch, useSelector } from "react-redux";

const Count = () => {
    const count = useSelector((state) => state.counter.value);
    const dispatch = useDispatch();

    const [newcount,setnewcount] = useState(0);

    return (
        <div>
            <h1>Count: {count}</h1>
            <button onClick={() => dispatch(increment())}>Increment</button>
            <button onClick={() => dispatch(decrement())}>Decrement</button>
            <input type="number" value={newcount} onChange={(e) => setnewcount(e.target.value)} />  
            <button onClick={() => dispatch(incrementByAmount(Number(newcount)))}>Increment by Amount</button>
            <button onClick={() => setnewcount(0)}>Reset</button>
        </div>
    );
};

export default Count;

