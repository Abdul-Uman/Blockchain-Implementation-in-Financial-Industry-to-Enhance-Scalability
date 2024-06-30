pragma solidity >= 0.8.11 <= 0.8.11;

contract FinancialContract {
    string public users;
    string public products;
    string public wallets;
    string public cart;

    function addUser(string memory us) public {
        users = us;	
    }

    function getUser() public view returns (string memory) {
        return users;
    }

    function addProducts(string memory p) public {
        products = p;	
    }

    function getProducts() public view returns (string memory) {
        return products;
    }

    function addWallets(string memory w) public {
        wallets = w;	
    }

    function getWallets() public view returns (string memory) {
        return wallets;
    }

    function addCart(string memory c) public {
        cart = c;	
    }

    function getCart() public view returns (string memory) {
        return cart;
    }

    constructor() public {
        users = "empty";
	products = "empty";
	wallets="empty";
	cart="empty";
    }
}