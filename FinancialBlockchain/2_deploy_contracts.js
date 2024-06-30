const FinancialContract = artifacts.require("FinancialContract");

module.exports = function(deployer) {
  deployer.deploy(FinancialContract);
};
